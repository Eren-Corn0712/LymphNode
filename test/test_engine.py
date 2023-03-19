from toolkit.engine.base_train import BaseTrainer
from toolkit.esvit.esvit_trainer import EsVitTrainer

DIR_DATASET = "../dataset"
DEFAULT_CFG = "../toolkit/cfg/default.yaml"
ESVIT_DEFAULT_CFG = "../toolkit/cfg/esvit_swim_tiny.yaml"


class TestClass(object):
    def __init__(self):
        pass

    def test_base_trainer(self, *args, **kwargs):
        base_trainer = BaseTrainer(DEFAULT_CFG)

    def test_esvit_trainer(self, *args, **kwargs):
        esvit_trainer = EsVitTrainer(ESVIT_DEFAULT_CFG)
        esvit_trainer.train()
    def __call__(self, *args, **kwargs):
        method = sorted(name for name in dir(self) if name.islower() and name.startswith("test"))
        for m in method:
            print(f"Test the method with endswith test_{m}")
            getattr(self, f"{m}")(*args, **kwargs)


TestClass()()
