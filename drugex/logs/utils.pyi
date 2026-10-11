from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, overload
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem

import datetime
import json
import logging
import os
import re
import shutil
import subprocess
import warnings
from drugex.logs import config
from drugex.logs import logger
from drugex.logs import setLogger
from drugex.logs.config import LogFileConfig

BACKUP_DIR_FOLDER_PREFIX: Any = ...

def export_conda_environment(filepath: str) -> None:
    """
    Export the conda environment to a yaml file.

    Args:
        filepath (str): path to the yaml file

    Raises:
        subprocess.CalledProcessError: if the command fails
        Exception: if an unexpected error occurs
    """

    ...


def enable_file_logger(log_folder: str, filename: str, debug: bool = ..., log_name: Optional[str] = ..., git_hash: Optional[str] = ..., init_data: Optional[Dict] = ..., disable_existing_loggers: bool = ...) -> LogFileConfig:
    ...


def generate_backup_runID(path: str = ...) -> int:
    """
    Generates runID for generation backups of files to be overwritten 
    If no previous backfiles (starting with #) exists, runid is set to 0, else to previous runid+1
    """

    ...


def generateBackupDir(root: str, backup_id: int) -> str:
    ...


def backUpFilesInFolder(_dir: str, backup_id: int, output_prefixes: Union[str, Tuple[str, ...]], output_extensions: Union[str, Tuple[str, ...]] = ..., cp_suffix: Optional[str] = ...) -> str:
    ...


def backUpFiles(base_dir: str, folder: str, output_prefixes: tuple, cp_suffix: Optional[str] = ...) -> str:
    ...


def callwarning(warning_text: str, exp: Optional[Type[Warning]] = ...) -> Callable:
    ...

