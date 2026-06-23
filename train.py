from libs.dataset.data import build_dataset, multibatch_collate_fn
from libs.dataset.transform import TrainTransform
from libs.utils.logger import AverageMeter
from libs.utils.loss import *
from libs.utils.utility import parse_args, save_checkpoint, adjust_learning_rate

import torch
torch.set_printoptions(profile='full')
import torch.optim as optim
import torch.utils.data as data
import libs.utils.logger as logger

import os
import os.path as osp
import copy

from progress.bar import Bar
from tensorboardX import SummaryWriter

import torch.nn.functional as F
from vivim import Vivim
MAX_FLT = 1e6
global_itnum = 1
opt, _ = parse_args()

device = 'cuda:{}'.format(opt.gpu_id)


class ForegroundBackgroundContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.5, use_cosine_similarity=True, memory_size=1024):
        super().__init__()
        self.temperature = temperature
        self.use_cosine_similarity = use_cosine_similarity
        self.memory_size = memory_size
        self.register_buffer('foreground_memory', None)
        self.register_buffer('background_memory', None)
        if use_cosine_similarity:
            self.similarity_function = torch.nn.CosineSimilarity(dim=-1)
        else:
            self.similarity_function = self._dot_similarity
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    @staticmethod
    def _dot_similarity(x, y):
        return torch.tensordot(x, y, dims=([1], [1]))

    def forward(self, foreground, background):
        batch_size = foreground.shape[0]
        foreground_flat = foreground.view(batch_size, -1)
        background_flat = background.view(batch_size, -1)

        if self._memory_is_full():
            loss = self._memory_contrastive_loss(foreground_flat, background_flat)
        else:
            loss = self._batch_contrastive_loss(foreground_flat, background_flat)
        self._update_memory(foreground_flat, background_flat)
        return loss

    def _batch_contrastive_loss(self, foreground_flat, background_flat):
        batch_size = foreground_flat.shape[0]
        positive_similarities = self.similarity_function(
            foreground_flat.unsqueeze(1), foreground_flat.unsqueeze(0)
        )
        negative_similarities = self.similarity_function(foreground_flat, background_flat)
        positive_mask = torch.eye(batch_size).bool().to(foreground_flat.device)
        positives = positive_similarities[positive_mask].view(batch_size, 1)
        negatives = negative_similarities.view(batch_size, 1)

        logits = torch.cat([positives, negatives], dim=1) / self.temperature
        labels = torch.zeros(batch_size).long().to(foreground_flat.device)
        loss = self.criterion(logits, labels)
        return loss / batch_size

    def _memory_contrastive_loss(self, foreground_flat, background_flat):
        foreground_bank = self.foreground_memory.to(foreground_flat.device)
        background_bank = self.background_memory.to(background_flat.device)
        background_bank = torch.cat([background_bank, background_flat], dim=0)

        positive_similarities = self.similarity_function(
            foreground_flat.unsqueeze(1), foreground_bank.unsqueeze(0)
        )
        negative_similarities = self.similarity_function(
            foreground_flat.unsqueeze(1), background_bank.unsqueeze(0)
        )

        positive_logits = torch.logsumexp(positive_similarities / self.temperature, dim=1, keepdim=True)
        negative_logits = negative_similarities / self.temperature
        logits = torch.cat([positive_logits, negative_logits], dim=1)
        labels = torch.zeros(foreground_flat.size(0)).long().to(foreground_flat.device)
        return self.criterion(logits, labels) / foreground_flat.size(0)

    def _memory_is_full(self):
        return self.foreground_memory is not None and self.foreground_memory.size(0) >= self.memory_size

    def _update_memory(self, foreground_flat, background_flat):
        foreground_flat = foreground_flat.detach()
        background_flat = background_flat.detach()

        if self.foreground_memory is None:
            self.foreground_memory = foreground_flat[-self.memory_size:].clone()
            self.background_memory = background_flat[-self.memory_size:].clone()
            return

        self.foreground_memory = torch.cat([self.foreground_memory, foreground_flat], dim=0)[-self.memory_size:].clone()
        self.background_memory = torch.cat([self.background_memory, background_flat], dim=0)[-self.memory_size:].clone()


def main():

    writer1 = SummaryWriter()
    use_gpu = torch.cuda.is_available() and int(opt.gpu_id) >= 0

    if not os.path.isdir(opt.checkpoint):
        os.makedirs(opt.checkpoint)

    opt.output = osp.join(osp.join(opt.checkpoint, opt.output_dir))
    if not osp.exists(opt.output):
        os.mkdir(opt.output)

    logfile = osp.join(opt.checkpoint, 'recurrent' + '_log.txt')
    logger.setup(filename=logfile, resume=opt.resume != '')
    log = logger.getLogger(__name__)

    log.info('Preparing dataset')

    input_dim = tuple(opt.input_size)
    train_transformer = TrainTransform(size=input_dim)

    datalist = []
    datalist_unlabel = []

    for dataset, freq, max_skip in zip(opt.trainset, opt.datafreq, opt.max_skip):

        ds = build_dataset(
            name=dataset,
            train=True,
            sampled_frames=opt.sampled_frames,
            transform=train_transformer,
            max_skip=max_skip,
            samples_per_video=opt.samples_per_video,
            datatype='label'
        )
        ds_unlabel = build_dataset(
            name=dataset,
            train=True,
            sampled_frames=opt.sampled_frames,
            transform=train_transformer,
            max_skip=max_skip,
            samples_per_video=opt.samples_per_video,
            datatype='unlabel'
        )
        datalist += [copy.deepcopy(ds) for _ in range(freq * 2)]
        datalist_unlabel += [copy.deepcopy(ds_unlabel) for _ in range(freq)]

    trainset = data.ConcatDataset(datalist)
    trainset_unlabel = data.ConcatDataset(datalist_unlabel)

    if opt.data_backend == 'PIL':
        trainloader = data.DataLoader(trainset, batch_size=opt.train_batch, shuffle=True, num_workers=opt.workers,
                                      collate_fn=multibatch_collate_fn)
        trainloader_unlabel = data.DataLoader(trainset_unlabel, batch_size=opt.train_batch, shuffle=True,
                                              num_workers=opt.workers, collate_fn=multibatch_collate_fn)
    else:
        raise TypeError('unkown data backend {}'.format(opt.data_backend))

    log.info("creating model")
    net_student = Vivim(pretrained_segformer=opt.pretrained_segformer)
    net_teacher = Vivim(pretrained_segformer=opt.pretrained_segformer)
    if use_gpu:
        net_student = net_student.to(device)
        net_teacher = net_teacher.to(device)

    for p in net_student.parameters():
        p.requires_grad = True
    for param in net_teacher.parameters():
        param.detach_()

    log.info('Model Total params: %.2fM' % (sum(p.numel() for p in net_student.parameters()) / 1000000.0))

    if opt.adapt_memory:
        log.info('If adaptive memory: {0} Adapt memory maxsize: {1}'.format(opt.adapt_memory, opt.memory_max_Clip))

    if opt.loss == 'bce':
        criterion = torch.nn.BCEWithLogitsLoss()
    elif opt.loss == 'dice':
        criterion = SoftDiceLoss()
    elif opt.loss == 'both':
        criterion = lambda batch_pred, batch_masks: (torch.nn.BCEWithLogitsLoss()(batch_pred, batch_masks)
                                                     + SoftDiceLoss()(batch_pred, batch_masks))
    else:
        raise TypeError('unknown training loss %s' % opt.loss)

    if opt.solver == 'sgd':
        optimizer = optim.SGD(net_student.parameters(), lr=opt.learning_rate,
                              momentum=opt.momentum[0], weight_decay=opt.weight_decay)
    elif opt.solver == 'adam':
        optimizer = optim.Adam(net_student.parameters(), lr=opt.learning_rate,
                               betas=opt.momentum, weight_decay=opt.weight_decay)
    else:
        raise TypeError('unkown solver type %s' % opt.solver)

    if opt.resume:
        log.info('Resuming from checkpoint {}'.format(opt.resume))
        assert os.path.isfile(opt.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(opt.resume, map_location=device)
        start_epoch = checkpoint['epoch']
        net_student.load_param(checkpoint['state_dict'])
        net_teacher.load_param(checkpoint['state_dict'])
        skips = checkpoint['max_skip']

        try:
            if isinstance(skips, list):
                for idx, skip in enumerate(skips):
                    trainset.datasets[idx].set_max_skip(skip)
            else:
                trainset.set_max_skip(skips[0])
        except:
            log.warn('Initializing max skip fail')

    else:
        start_epoch = 0

    for epoch in range(start_epoch):
        adjust_learning_rate(optimizer, epoch, opt)

    contrastive_loss_fn = ForegroundBackgroundContrastiveLoss(memory_size=opt.contrastive_memory_size)
    contrastive_loss_fn.to(device)

    for epoch in range(start_epoch, opt.epochs):

        log.info('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, opt.epochs, opt.learning_rate))
        adjust_learning_rate(optimizer, epoch, opt)

        log.info('Skip Info:')
        skip_info = dict()
        if isinstance(trainset, data.ConcatDataset):
            for dataset in trainset.datasets:
                skip_info.update(
                    {type(dataset).__name__: dataset.max_skip}
                )
        else:
            skip_info.update(
                {type(trainset).__name__: dataset.max_skip}
            )

        skip_print = ''
        for k, v in skip_info.items():
            skip_print += '{}: {} '.format(k, v)
        log.info(skip_print)
        train_loss = train(trainloader,
                           trainloader_unlabel,
                           model_student=net_student,
                           model_teacher=net_teacher,
                           criterion=criterion,
                           optimizer=optimizer,
                           epoch=epoch,
                           use_cuda=use_gpu,
                           contrastive_loss_fn=contrastive_loss_fn)

        skips = [ds.max_skip for ds in trainset.datasets]
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net_student.state_dict(),
            'max_skip': skips,
        }, epoch + 1, checkpoint=opt.checkpoint)

        log_format = 'Epoch: {} LR: {} Loss: {}'
        log.info(log_format.format(epoch + 1, opt.learning_rate, train_loss))

        writer1.add_scalar('train_loss', train_loss, global_step=epoch)

        if (epoch + 1) % opt.epochs_per_increment == 0:
            if isinstance(trainset, data.ConcatDataset):
                for dataset in trainset.datasets:
                    dataset.increase_max_skip()
            else:
                trainset.increase_max_skip()


def train(trainloader, trainloader_unlabel, model_student, model_teacher, criterion, optimizer, epoch, use_cuda, contrastive_loss_fn):
    global global_itnum
    data_time = AverageMeter()
    bar = Bar('Processing', max=len(trainloader))
    total_loss = 0.0
    total_unlabel_loss = 0.0
    total_mseloss = 0.0
    total_contrastive_loss1 = 0.0
    total_contrastive_loss2 = 0.0

    for (batch_idx, (label_data, unlabel_data)) in enumerate(zip(trainloader, trainloader_unlabel)):

        global_itnum = global_itnum + 1

        label_loss, _, _, _, contrastive_loss_label = calculate_label(model_student, use_cuda, label_data, criterion, contrastive_loss_fn)
        unlabel_loss, mse_loss, contrastive_loss_unlabel = calculate_unlabel(model_student, model_teacher, use_cuda, unlabel_data, criterion, contrastive_loss_fn)
        contrastive_loss = contrastive_loss_label + contrastive_loss_unlabel
        batch_loss = label_loss + unlabel_loss + mse_loss + contrastive_loss

        total_loss = total_loss + batch_loss
        total_unlabel_loss = total_unlabel_loss + unlabel_loss
        total_mseloss = total_mseloss + mse_loss
        total_contrastive_loss1 = total_contrastive_loss1 + contrastive_loss

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        ema_decay = 0.9
        update_ema_variables(model_student, model_teacher, ema_decay, global_itnum)

        bar.suffix = ('(Epoch: {epoch} {batch}/{size}) Data: {data:.3f}s |Loss: {loss:.5f} | UnLoss: {unloss: .5f} '
                      '|MSE_Loss: {mseloss:.5f} |Contrastive_Loss1: {conloss:.5f} ').format(
            epoch=epoch,
            batch=batch_idx + 1,
            size=len(trainloader),
            data=data_time.val,
            loss=total_loss.item(),
            unloss=total_unlabel_loss.item(),
            mseloss=total_mseloss.item(),
            conloss=total_contrastive_loss1.item()
        )
        bar.next()
    bar.finish()

    return total_loss.item()


def calculate_label(model_student, use_cuda, label_data, criterion, contrastive_loss_fn):

    batch_frames, batch_masks, objs, batch_detail = label_data
    if use_cuda:
        batch_frames = batch_frames.to(device)
        batch_masks = batch_masks.to(device)

    predict_label, feature, feature2 = model_student(batch_frames)

    feature2 = torch.mean(feature2, dim=1, keepdim=True)
    feature2 = F.interpolate(feature2, size=(256, 256), mode='bilinear', align_corners=False)
    mask = batch_masks.flatten(start_dim=0, end_dim=1)
    foreground_feature = feature2 * mask
    background_feature = feature2 * (1-mask)
    foreground_feature_fill = fill_feature_map_with_neighbors(foreground_feature, 4)
    background_feature_fill = fill_feature_map_with_neighbors(background_feature, 4)

    contrastive_loss = contrastive_loss_fn(foreground_feature_fill, background_feature_fill)
    batch_loss = criterion(predict_label, batch_masks.flatten(start_dim=0, end_dim=1))

    return batch_loss, predict_label, feature, feature2, contrastive_loss


def calculate_unlabel(model_student, model_teacher, use_cuda, unlabel_data, criterion, contrastive_loss_fn):

    batch_frames, batch_masks, objs, batch_detail = unlabel_data

    loss1, predict_student, feature_student, feature_student2, contrastive_loss = calculate_label(model_student, use_cuda, unlabel_data, criterion, contrastive_loss_fn)

    noise = torch.clamp(torch.randn_like(batch_frames) * 0.1, -0.5, 0.5)
    batch_frames = batch_frames + noise
    if use_cuda:
        batch_frames = batch_frames.to(device)
        batch_masks = batch_masks.to(device)

    predict_teacher, feature_teacher, feature_teacher2 = model_teacher(batch_frames)

    mse_loss = torch.abs(F.mse_loss(torch.sigmoid(predict_student), torch.sigmoid(predict_teacher)))

    return loss1, mse_loss, contrastive_loss


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def fill_feature_map_with_neighbors(feature, N):
    B, _, H, W = feature.shape
    unfolded = F.unfold(feature, kernel_size=N, stride=N)

    unfolded = unfolded.view(B, 1, N * N, -1)

    filled_feature = feature.clone()

    for i in range(B):
        for block_idx in range(unfolded.shape[2]):
            block = unfolded[i, 0, :, block_idx]

            if (block == 0).any():
                non_zero_values = block[block != 0]

                if non_zero_values.numel() > 0:
                    replacement_value = non_zero_values[torch.randint(0, non_zero_values.numel(), (1,)).item()]
                    block[block == 0] = replacement_value
                else:
                    pass

            y_start = (block_idx // (W // N)) * N
            x_start = (block_idx % (W // N)) * N

            filled_feature[i, 0, y_start:y_start + N, x_start:x_start + N] = block.view(N, N)

    return filled_feature
if __name__ == '__main__':
    main()
