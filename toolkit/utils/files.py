import contextlib
import glob
import os
import urllib
import errno


from datetime import datetime
from pathlib import Path
from typing import List


def find_files(root: str, fmt: str = "png", recursive: bool = False) -> List[Path]:
    # Return all files with the specified format in the directory.
    pattern = f'**/*.{fmt}' if recursive else f'*.{fmt}'
    return sorted(list(Path(root).glob(pattern=pattern)))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
