import contextlib
import hashlib
import json
import os
import subprocess
import time
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tarfile import is_tarfile
from typing import List
from zipfile import is_zipfile

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageOps
from tqdm import tqdm

HELP_URL = 'See https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'webm'  # video suffixes
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders
IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def find_files(root: str, fmt: str = "png", recursive: bool = False) -> List[Path]:
    # Return all files with the specified format in the directory.
    pattern = f'**/*.{fmt}' if recursive else f'*.{fmt}'
    return sorted(list(Path(root).glob(pattern=pattern)))
