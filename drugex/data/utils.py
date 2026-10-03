"""Path resolution and data discovery utilities for DrugEx datasets."""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

from drugex import VERSION
from drugex.logs import logger


def getVocPaths(data_path: str, voc_files: Sequence[str], mol_type: str) -> List[str]:
    """Resolve candidate vocabulary file paths against a data directory, logging warnings on misses."""
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
    mol_type: str,
    unique_frags: bool
) -> Tuple[str, str]:
    """Resolve training and test dataset file paths based on input prefix, format, and fragment flags."""
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
