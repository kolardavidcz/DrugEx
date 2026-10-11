"""
Utility functions to download files.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, overload
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem

import os
import hashlib
import zipfile
import tarfile
import requests
from tqdm.auto import tqdm

CHUNKSIZE: Any = ...
RETRIES: Any = ...

def download_file(url: str, out_path: str, extract_out_path: Optional[str] = ..., byte_size: Optional[int] = ..., sha256sum: Optional[str] = ..., progress: bool = ..., callback: Optional[Callable] = ...) -> None:
    """
    Download a file and extract its content if is a ZIP or TAR.GZ file.

    Args:
        url (str): URL of the file to be downloaded.
        out_path (str): Path the file should be written to
        extract_out_path (str): Path to extract the content of the ZIP/TAR.GZ file into
        byte_size (int): Size in bytes of the file to be downloaded; ignored if None
        sha256sum (str): SHA256 checksum to compare that of the downloaded file to
        progress (bool): should progress be shown
        callback (Callable[[], Any]): callback function to be called after each chunk of the file is downloaded
    """

    ...


def check_sha256sum(filename: str, sha256: str) -> bool:
    """
    Check the SHA256 checksum of a file corresponds to that given.

    Args:
        filename (str): Path the file to check
        sha256 (str): SHA256 checksum to compare that of the downloaded file to

    Returns:
        True if the file's SHA256 checksum and the one provided match
    """

    ...


def sha256sum(filename: str, blocksize: Optional[int] = ...) -> str:
    """
    Calculate the SHA256 cheksum of a file.

    Args:
        filename: path of the file
        blocksize: size of iterative chunks for calculation (default: 64 KiB)

    Returns:
        SHA256 checksum is hexadecimal
    """

    ...

