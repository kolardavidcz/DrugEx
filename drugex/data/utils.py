"""Path resolution and data discovery utilities for DrugEx datasets."""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

from drugex import VERSION
from drugex.logs import logger


def getVocPaths(data_path: str, voc_files: Sequence[str], mol_type: str = 'smiles') -> List[str]:
    """Retrieve filesystem paths to vocabulary files, logging fallbacks if absent.

    Parameters
    ----------
    data_path : str
        Base path to the data directory.
    voc_files : sequence of str
        List of candidate vocabulary file names.
    mol_type : str, optional
        Representation format (`'smiles'` or `'graph'`) used for suffixed fallbacks (default: `'smiles'`).

    Returns
    -------
    voc_paths : list of str
        List of resolved, existing vocabulary file paths.
    """
    voc_paths: List[str] = []
    for voc_file in voc_files:
        path = f'{data_path}/{voc_file}'
        if os.path.exists(path):
            voc_paths.append(path)
        elif os.path.exists(path + f'_{mol_type}.txt.vocab'):
            voc_paths.append(path + f'_{mol_type}.txt.vocab')
        else:
            logger.warning(f'Could not find vocabulary file {voc_file} in {data_path}.')

    if len(voc_paths) == 0:
        logger.warning(f'No vocabulary files found. Using internal defaults for DrugEx v{VERSION}.')

    return voc_paths


def getDataPaths(
    data_path: str,
    input_prefix: str,
    mol_type: str = 'smiles',
    unique_frags: bool = False
) -> Tuple[str, str]:
    """Resolve filesystem paths for training and validation datasets.

    Parameters
    ----------
    data_path : str
        Base directory containing dataset files.
    input_prefix : str
        Prefix string identifying dataset files.
    mol_type : str, optional
        Representation format (`'smiles'` or `'graph'`) (default: `'smiles'`).
    unique_frags : bool, optional
        If True, targets unique fragment splits rather than standard training splits (default: False).

    Returns
    -------
    paths : tuple of (str, str)
        Resolved `(train_path, test_path)`.

    Raises
    ------
    AssertionError
        If resolved train or test file paths do not exist on disk.
    """
    if os.path.exists(data_path + input_prefix):
        train_path = data_path + input_prefix
        test_path = train_path
    else:
        train_path = data_path + '_'.join([input_prefix, 'unique' if unique_frags else 'train', mol_type]) + '.txt'
        test_path = data_path + '_'.join([input_prefix, 'test', mol_type]) + '.txt'

    assert os.path.exists(train_path), f'{train_path} does not exist'
    assert os.path.exists(test_path), f'{test_path} does not exist'

    logger.info(f'Loading training data from {train_path}')
    logger.info(f'Loading validation data from {test_path}')

    return train_path, test_path
