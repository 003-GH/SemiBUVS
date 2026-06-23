from yacs.config import CfgNode

OPTION = CfgNode()

OPTION.trainset = ['VOS']
OPTION.valset = 'VOS'
OPTION.data_root = 'Youtube-VOS-CL'
OPTION.datafreq = [1]
OPTION.input_size = (256, 256)   # input image size for training
OPTION.sampled_frames = 3        # min sampled time length while trianing
OPTION.max_skip = [3]       # max skip time length while trianing
OPTION.samples_per_video = 1    # sample numbers per video
OPTION.data_backend = 'PIL'     # dataloader backend 'PIL'

OPTION.epochs_per_increment = 5

OPTION.epochs = 100
OPTION.train_batch = 4
OPTION.learning_rate = 0.0001
OPTION.gamma = 1
OPTION.momentum = (0.9, 0.999)
OPTION.solver = 'adam'             # 'sgd' or 'adam'
OPTION.weight_decay = 0
OPTION.milestone = [int(999)]     # epochs to degrades the learning rate
OPTION.loss = 'both'               # 'bce' or 'dice' or 'both'

OPTION.epoch_per_test = 1
OPTION.correction_rate = 150
OPTION.correction_momentum = 0.9
OPTION.loop = 10


OPTION.adapt_memory = True
OPTION.memory_max_Clip = 21
OPTION.contrastive_memory_size = 32


OPTION.max_sample_frames_training = 3

OPTION.checkpoint = 'ckpt'
OPTION.initial = ''       # path to initialize the backbone
OPTION.resume = ''       # path to restart from the checkpoint
OPTION.video_path = ''   # path to video on which the model is running
OPTION.mask_path = ''    # path to mask on withc the model is running
OPTION.pretrained_segformer = 'segformer-b3-finetuned-ade-512-512'
OPTION.gpu_id = 0
OPTION.workers = 1
OPTION.save_indexed_format = 'index'
OPTION.output_dir = 'output'
OPTION.test_checkpoint_dir = 'ckpt'

OPTION.multi_gpu_ids = [4]
OPTION.backend = 'nccl'
OPTION.init_method = 'env://'


def sanity_check(opt):

    assert isinstance(opt.trainset, (str, list)), \
        'training set should be specified by a string or string list'
    assert isinstance(opt.valset, str), \
        'validation set should be a single dataset'
    assert opt.data_backend in ['PIL'], \
        'only PIL backend are supported'
    assert opt.solver in ['adam', 'sgd']
    assert opt.loss in ['dice', 'bce', 'both']


def getCfg():
    return OPTION.clone()
