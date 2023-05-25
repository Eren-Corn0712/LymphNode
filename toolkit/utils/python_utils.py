import torch
import numpy as np

from numpy import ndarray
from torch import Tensor


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


def merge_dict_with_prefix(a, b, prefix='', include=(), exclude=()):
    """
    Merge dictionary `b` into dictionary `a`, adding a prefix to the keys in `b` and
    optionally including or excluding specific keys.

    Args:
        a (dict): The dictionary to merge into.
        b (dict): The dictionary to merge from.
        prefix (str, optional): The prefix to add to the keys in `b`. Defaults to ''.
        include (tuple, optional): Tuple of keys to include. Defaults to ().
        exclude (tuple, optional): Tuple of keys to exclude. Defaults to ().

    Returns:
        dict: The updated dictionary `a`.

    """
    if not include:
        include = b.keys()

    for k, v in b.items():
        if k in include and k not in exclude:
            a[f'{prefix}{k}'] = v

    return a


def batch_dataconcat(a, b):
    if a == {}:
        return b

    for (key1, value1), (key2, value2) in zip(a.items(), b.items()):
        assert key1 == key2, "Two dictionary have different key"
        if isinstance(value1, list) and isinstance(value2, list):
            value1 += value2
        elif isinstance(value1, tuple) and isinstance(value2, tuple):
            value1 += value2
        elif isinstance(value1, Tensor) and isinstance(value2, Tensor):
            value1 = torch.cat([value1, value2], dim=0)
        elif isinstance(value1, ndarray) and isinstance(value2, ndarray):
            value1 = np.concatenate([value1, value2], axis=0)
        else:
            raise ValueError(f"isinstance of {type(value1)} is not supported.")

        a[key1] = value1
    return a
