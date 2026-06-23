from libs.dataset.data import ROOT, build_dataset, multibatch_collate_fn
from libs.dataset.transform import TestTransform
from libs.utils.logger import AverageMeter
from libs.utils.utility import parse_args, write_mask

import torch
import torch.utils.data as data
import libs.utils.logger as logger
import os
import time

from progress.bar import Bar

from vivim import Vivim

opt, _ = parse_args()

device = 'cuda:{}'.format(opt.gpu_id)
use_gpu = torch.cuda.is_available() and int(opt.gpu_id) >= 0

logger.setup(filename='test_out.log', resume=False)
log = logger.getLogger(__name__)

def main(model_name='', model_path=''):

    log.info('Preparing dataset %s' % opt.valset)

    input_dim = opt.input_size

    test_transformer = TestTransform(size=input_dim)
    testset = build_dataset(
        name=opt.valset,
        train=False, 
        transform=test_transformer, 
        samples_per_video=1
        )

    testloader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=opt.workers,
                                collate_fn=multibatch_collate_fn)
    log.info("Creating model")

    if opt.adapt_memory:
        log.info('If adaptive memory: {0} Adapt memory maxsize: {1}'.format(opt.adapt_memory, opt.memory_max_Clip))

    net = Vivim(pretrained_segformer=opt.pretrained_segformer)
    log.info('Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))

    net.eval()

    if use_gpu:
        net.to(device)

    for p in net.parameters():
        p.requires_grad = False

    if os.path.isfile(opt.initial):
        log.info('Loading weights from checkpoint {}'.format(opt.initial))
        assert os.path.isfile(opt.initial), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(opt.initial, map_location=device)
        try:
            net.load_param(checkpoint['state_dict'])
        except:
            net.load_param(checkpoint)
    elif os.path.isfile(model_path):
        log.info('Loading weights from checkpoint {}'.format(model_path))
        assert os.path.isfile(model_path), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(model_path, map_location=device)
        try:
            net.load_param(checkpoint['state_dict'])
        except:
            net.load_param(checkpoint)

    log.info('Runing model on dataset {}, totally {:d} videos'.format(opt.valset, len(testloader)))
    
    test_adaptive_memory(
                        testloader,
                        model=net,
                        use_cuda=use_gpu,
                        opt=opt,
                        model_name=model_name)

    log.info('Results are saved at: {}'.format(os.path.join(ROOT, opt.output_dir, opt.valset)))


def test_adaptive_memory(testloader, model, use_cuda, model_name, opt):
    data_time = AverageMeter()
    bar = Bar('Processing model testing', max=len(testloader))

    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):

            frames, masks, objs, infos = data

            if use_cuda:
                frames = frames.squeeze(0).to(device)
                masks = masks.squeeze(0).to(device)

            info = infos[0]

            t1 = time.time()

            frames = frames.unsqueeze(0)

            for test_tag in range(0, len(objs) - 1):
                if not objs[test_tag] == objs[test_tag + 1]:
                    raise AssertionError('unexpected object count')

            num_objects = objs[0]
            pred, _, _ = model(frames)
            pred = pred.detach().cpu().numpy()

            assert num_objects == 1

            write_mask(pred, info, opt, directory=opt.output_dir, model_name='{}'.format(model_name))

            toc = time.time() - t1
            data_time.update(toc, 1)

            bar.suffix = '({batch}/{size}) Time: {data:.3f}s'.format(
                batch=batch_idx + 1,
                size=len(testloader),
                data=data_time.sum
            )
            bar.next()
        bar.finish()

    return data_time.sum


if __name__ == '__main__':

    models_path = opt.test_checkpoint_dir
    models = [file for file in os.listdir(models_path) if file.endswith('pth.tar')]
    for model_name in models:
        model_path = os.path.join(models_path, model_name)
        model_name = model_name.replace('.', '_')
        main(model_name=model_name, model_path=model_path)
