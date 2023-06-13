import os

from main_esvit_lymph import esvit
# from eval_linear import eval_linear
from eval_linear_recycle import eval_linear_recyle
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ESVIT_CFG = "toolkit/cfg/args.yaml"
LINEAR_CFG1 = "toolkit/cfg/args_linear.yaml"
LINEAR_CFG2 = "args_linear1.yaml"

if __name__ == "__main__":
    esvit(ESVIT_CFG)
    eval_linear_recyle(LINEAR_CFG1)
