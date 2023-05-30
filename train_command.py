import os

from main_esvit_lymph import esvit
from eval_linear import eval_linear

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
CFG1 = "args.yaml"
CFG2 = "args_linear.yaml"

if __name__ == "__main__":
    esvit(CFG1)
    eval_linear(CFG2)
