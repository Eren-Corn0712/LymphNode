import contextlib
import inspect
import logging.config
import os
import platform
import re
import argparse
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Union
from collections import defaultdict

import cv2
import numpy as np
import torch
import yaml

# PyTorch Multi-GPU DDP Constants
RANK = int(os.getenv('RANK', -1))
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

# Constants
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # project root
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLOv5 multiprocessing threads
AUTOINSTALL = str(os.getenv('YOLO_AUTOINSTALL', True)).lower() == 'true'  # global auto-install mode
VERBOSE = str(os.getenv('YOLO_VERBOSE', True)).lower() == 'true'  # global verbose mode
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format
LOGGING_NAME = 'toolkit'
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ['Darwin', 'Linux', 'Windows'])  # environment booleans

# Settings
torch.set_printoptions(linewidth=320, precision=4, profile='default')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ['NUMEXPR_MAX_THREADS'] = str(NUM_THREADS)  # NumExpr max threads
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # for deterministic training
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress verbose TF compiler warnings in Colab


class SimpleClass:
    """
    Ultralytics SimpleClass is a base class providing helpful string representation, error reporting, and attribute
    access methods for easier debugging and usage.
    """

    def __str__(self):
        """Return a human-readable string representation of the object."""
        attr = []
        for a in dir(self):
            v = getattr(self, a)
            if not callable(v) and not a.startswith('_'):
                if isinstance(v, SimpleClass):
                    # Display only the module and class name for subclasses
                    s = f'{a}: {v.__module__}.{v.__class__.__name__} object'
                else:
                    s = f'{a}: {repr(v)}'
                attr.append(s)
        return f'{self.__module__}.{self.__class__.__name__} object with attributes:\n\n' + '\n'.join(attr)

    def __repr__(self):
        """Return a machine-readable string representation of the object."""
        return self.__str__()

    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")


class IterableSimpleNamespace(SimpleNamespace):
    """
    Ultralytics IterableSimpleNamespace is an extension class of SimpleNamespace that adds iterable functionality and
    enables usage with dict() and for loops.
    """

    def __iter__(self):
        """Return an iterator of key-value pairs from the namespace's attributes."""
        return iter(vars(self).items())

    def __str__(self):
        """Return a human-readable string representation of the object."""
        return '\n'.join(f'{k}={v}' for k, v in vars(self).items())

    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(f"""
            '{name}' object has no attribute '{attr}'. This may be caused by a modified or out of date ultralytics
            'default.yaml' file.\nPlease update your code with 'pip install -U ultralytics' and if necessary replace
            {DEFAULT_CFG_PATH} with the latest version from
            https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/cfg/default.yaml
            """)

    def get(self, key, default=None):
        """Return the value of the specified key if it exists; otherwise, return the default value."""
        return getattr(self, key, default)


def plt_settings(rcparams={'font.size': 11}, backend='Agg'):
    """
    Decorator to temporarily set rc parameters and the backend for a plotting function.

    Usage:
        decorator: @plt_settings({"font.size": 12})
        context manager: with plt_settings({"font.size": 12}):

    Args:
        rcparams (dict): Dictionary of rc parameters to set.
        backend (str, optional): Name of the backend to use. Defaults to 'Agg'.

    Returns:
        callable: Decorated function with temporarily set rc parameters and backend.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            original_backend = plt.get_backend()
            plt.switch_backend(backend)

            with plt.rc_context(rcparams):
                result = func(*args, **kwargs)

            plt.switch_backend(original_backend)
            return result

        return wrapper

    return decorator


def set_logging(name=LOGGING_NAME, verbose=True):
    # sets up logging for the given name
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            name: {
                'format': '%(message)s'}},
        'handlers': {
            name: {
                'class': 'logging.StreamHandler',
                'formatter': name,
                'level': level}},
        'loggers': {
            name: {
                'level': level,
                'handlers': [name],
                'propagate': False}}})


class EmojiFilter(logging.Filter):
    """
    A custom logging filter class for removing emojis in log messages.

    This filter is particularly useful for ensuring compatibility with Windows terminals
    that may not support the display of emojis in log messages.
    """

    def filter(self, record):
        record.msg = emojis(record.msg)
        return super().filter(record)


# Set logger
set_logging(LOGGING_NAME, verbose=VERBOSE)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)
if WINDOWS:  # emoji-safe logging
    LOGGER.addFilter(EmojiFilter())


def yaml_save(file='data.yaml', data=None):
    """
    Save YAML data to a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        data (dict, optional): Data to save in YAML format. Default is None.

    Returns:
        None: Data is saved to the specified file.
    """
    file = Path(file)
    if not file.parent.exists():
        # Create parent directories if they don't exist
        file.parent.mkdir(parents=True, exist_ok=True)

    with open(file, 'w') as f:
        # Dump data to file in YAML format, converting Path objects to strings
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v
                        for k, v in data.items()},
                       f,
                       sort_keys=False,
                       allow_unicode=True)


def yaml_load(file='data.yaml', append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        dict: YAML data and file name.
    """
    with open(file, errors='ignore', encoding='utf-8') as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)

        # Add YAML filename to dict and return
        return {**yaml.safe_load(s), 'yaml_file': str(file)} if append_filename else yaml.safe_load(s)


def yaml_print(yaml_file: Union[str, Path, dict]) -> None:
    """
    Pretty prints a yaml file or a yaml-formatted dictionary.

    Args:
        yaml_file: The file path of the yaml file or a yaml-formatted dictionary.

    Returns:
        None
    """
    yaml_dict = yaml_load(yaml_file) if isinstance(yaml_file, (str, Path)) else yaml_file
    dump = yaml.dump(yaml_dict, sort_keys=False, allow_unicode=True)
    LOGGER.info(f"Printing '{colorstr('bold', 'black', yaml_file)}'\n\n{dump}")


def emojis(string=''):
    # Return platform-dependent emoji-safe version of string
    return string.encode().decode('ascii', 'ignore') if WINDOWS else string


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


class TryExcept(contextlib.ContextDecorator):
    # YOLOv8 TryExcept class. Usage: @TryExcept() decorator or 'with TryExcept():' context manager
    def __init__(self, msg='', verbose=True):
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        pass

    def __exit__(self, exc_type, value, traceback):
        if self.verbose and value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
        return True


def threaded(func):
    # Multi-threads a target function and returns thread. Usage: @threaded decorator
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread

    return wrapper


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def print_options(opt):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    LOGGER.info(message)


def average_classification_reports(reports):
    # 初始化一個字典來儲存平均值
    averages = {}

    # 計算每個標籤的 metric 值總和
    for report in reports:
        for label, metrics in report.items():
            # init dict
            if label not in averages and isinstance(metrics, dict):
                averages[label] = {metric: 0 for metric in metrics}
            elif label not in averages and isinstance(metrics, float):
                averages[label] = 0

            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    averages[label][metric] += value
            elif isinstance(metrics, float):
                averages[label] += metrics

    # 計算平均值
    num_reports = len(reports)
    for label, metrics in averages.items():
        if isinstance(metrics, dict):
            for metric, value in metrics.items():
                averages[label][metric] = value / num_reports
        elif isinstance(metrics, float):
            averages[label] /= num_reports
    return averages
