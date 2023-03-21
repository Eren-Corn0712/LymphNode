import os
import torch
import torch.distributed as dist

import subprocess
from pathlib import Path
from toolkit.utils.checks import check_file, check_imgsz, print_args
from toolkit.cfg import get_cfg
from toolkit.utils import (DEFAULT_CFG, LOGGER, ONLINE, RANK, ROOT, SETTINGS, TQDM_BAR_FORMAT,
                           yaml_save, yaml_print, yaml_load)
from toolkit.utils.files import get_latest_run, increment_path
from toolkit.utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, init_seeds, one_cycle,
                                       select_device, strip_optimizer)
from toolkit.utils.dist import ddp_cleanup, generate_ddp_command


class BaseTrainer(object):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        self.args = get_cfg(cfg, overrides)
        self.device = select_device(self.args.device, self.args.batch)
        self.console = LOGGER
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)
        self.resume = False

        # Dirs
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = self.args.name or f'{self.args.mode}'
        if hasattr(self.args, 'save_dir'):
            self.save_dir = Path(self.args.save_dir)
        else:
            self.save_dir = Path(
                increment_path(Path(project) / name, exist_ok=self.args.exist_ok if RANK in (-1, 0) else True))

        self.wdir = self.save_dir / 'weights'  # weights dir
        if RANK in (-1, 0):
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / 'args.yaml', vars(self.args))  # save run args

        self.last, self.best = self.wdir / 'last.pt', self.wdir / 'best.pt'  # checkpoint paths

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device.type == 'cpu':
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        self.trainset, self.testset = None, None

        self.optimizer = None
        self.scheduler = None

        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.csv = self.save_dir / 'results.csv'

    def train(self):
        # Allow device='', device=None on Multi-GPU systems to default to device=0
        if isinstance(self.args.device, int) or self.args.device:  # i.e. device=0 or device=[0,1,2,3]
            world_size = torch.cuda.device_count()
        elif torch.cuda.is_available():  # i.e. device=None or device=''
            world_size = 1  # default to device 0
        else:  # i.e. device='cpu' or 'mps'
            world_size = 0

        # Run subprocess if DDP training, else train normally
        if world_size > 1 and 'LOCAL_RANK' not in os.environ:
            cmd, file = generate_ddp_command(world_size, self)  # security vulnerability in Snyk scans
            try:
                LOGGER.info(f'Running DDP command {cmd}')
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))
        else:
            self._do_train(RANK, world_size)

    def _setup_ddp(self, rank, world_size):
        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '9020'
        torch.cuda.set_device(rank)
        self.device = torch.device('cuda', rank)
        LOGGER.info(f'DDP settings: RANK {rank}, WORLD_SIZE {world_size}, DEVICE {self.device}')
        dist.init_process_group('nccl' if dist.is_nccl_available() else 'gloo', rank=rank, world_size=world_size)

    def _setup_train(self, rank, world_size):
        raise NotImplementedError

    def _do_train(self, rank=-1, world_size=1):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError

    def get_dataset(self):
        raise NotImplementedError

    def setup_model(self):
        raise NotImplementedError

    def optimizer_step(self):
        raise NotImplementedError

    def preprocess_batch(self, batch):
        return batch

    def get_model(self, cfg=None, weights=None, verbose=True):
        raise NotImplementedError("Please implement get_model method.")

    def get_dataloader(self):
        raise NotImplementedError("get_dataloader function not implemented in trainer.")

    def criterion(self, preds, batch):
        raise NotImplementedError("criterion function not implemented in trainer.")

    def label_loss_items(self, loss_items=None, prefix='train'):
        return {'loss': loss_items if loss_items is not None else ['loss']}

    def resume_training(self, ckpt):
        raise NotImplementedError("resume_training function not implemented")

    def progress_string(self):
        return ""

    def build_optimizer(self):
        raise NotImplementedError("build_optimizer function not implemented")
