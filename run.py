"""Entry point for evaluating Video-LLMs on the EgoExo-Con benchmark.

If you hit a pydantic error, pin the version with: pip install pydantic==1.10.8
"""

import torch
import random
import numpy as np
from utils.cons_utils import BaseOptions
from task.egoexo_consistency_batch import run_egoexo_consistency


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)

    print("Set seed ", seed)


if __name__ == "__main__":
    """Example:
        python run.py --model_type qwen2_5 --task grounding --debug
    """
    base_options = BaseOptions().parse()
    set_seed(base_options.seed)

    if "qwen2_5" in base_options.model_type:
        from baselines.qwen_2_5_vl import Qwen2_5_VL, Qwen2_5_VL_Options
        args = Qwen2_5_VL_Options().parse()
        model = Qwen2_5_VL(args)

    else:
        raise NotImplementedError

    run_egoexo_consistency(model=model, args=args)

