from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Generic, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, overload
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem

import git
import logging
import os
from bisect import bisect
from datetime import datetime
from logging import config
from typing import Dict


class LevelFilter(logging.Filter):
    """LoggingFilter used to filter one or more specific log levels messages"""

    def __init__(self, level: List[int]) -> None:
        ...

    def filter(self, record: logging.LogRecord) -> bool:
        """Returns True if record.levelno is in allowed levels list."""

        ...


class LevelFormatter(logging.Formatter):
    """LoggingFormatter used to specifiy the formatting per level"""

    def __init__(self, formats: Dict[int, str], **kwargs) -> None:
        ...

    def format(self, record: logging.LogRecord) -> str:
        """Renders log record using formatter corresponding to its level."""

        ...


def config_logger(log_file_path: str, debug: Optional[bool] = ..., disable_existing_loggers: bool = ...) -> None:
    """
    Function to configure the logging. 
    All info is saved in a simple format on the log file path.
    Debug entries are saved to a separate file if debug is True
    Debug and warning and above are save in a verbose format.
    Warning and above are also printed to std.out
    ...

    Arguments
    ----------
    log_file_path (str): Folder where all logs for this run are saved
    debug (bool): if true, debug messages are saved
    no_exist_log (bool): if true, existing loggers are disabled
    """

    ...


def init_logfile(log: logging.Logger, githash: Optional[str] = ..., args: Optional[Any] = ...) -> None:
    """
    Put some intial information in the logfile
    ...

    Arguments
    ----------
    log : Logging instance
    runid (str): the current runid
    githash (str): githash
    """

    ...


def get_git_info() -> None:
    """
    Get information of the current git commit

    If the package is installed with pip, read detailed version extracted by setuptools_scm.
    Otherwise, use gitpython to get the information from the git repo.
    """

    ...


def get_runid(log_folder: str = ..., old: bool = ..., id: Optional[int] = ...) -> str:
    """
    Fetch runid that is used in all logfiles to identifiy a specific run
    ...

    Arguments
    ----------
    log_folder (str): Folder where all logs are saved
    old (bool): if true, fetches the last used runid
    id (int): If included, returns this runid number
    """

    ...


class LogFileConfig:
    """
    Configuration container holding target log file directory path, active module logger reference, and debug flag for file logger initialization.

    Parameters
    ----------
    path : str
        Filesystem destination path for log files.
    logger : logging.Logger
        Target logging.Logger instance to configure.
    debug : bool
        If True, configures separate debug.log trace output (default: False).

    Notes
    -----
    Passed to enable_file_logger and config_logger to direct run output to runid folders.
    """

    def __init__(self, path: str, logger: logging.Logger, debug: bool) -> None:
        ...

