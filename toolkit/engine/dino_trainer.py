from toolkit.engine.base_train import BaseTrainer
from torchvision.models.swin_transformer import s

class DINOTrainer(BaseTrainer):
    def __init__(self, cfg, overrides=None):
        super(DINOTrainer, self).__init__(cfg=cfg, overrides=overrides)
