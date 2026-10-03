# -*- coding: utf-8 -*-

"""Utility functions to download files."""

import os
import hashlib
import zipfile
import tarfile

# [PACKAGE: requests] WHY: Remote HTTP streaming WHAT: Fetches files over HTTP with chunk streaming and session retry
import requests
# [PACKAGE: tqdm] WHY: CLI progress visualization WHAT: Renders interactive transfer progress bars
from tqdm.auto import tqdm


CHUNKSIZE = 1048576  # 1 MB
RETRIES = 3


def download_file(url, out_path, extract_out_path=None, byte_size=None, sha256sum=None, progress=True, callback=None) -> None:
    """Download a file with retry and optional checksum validation, unpacking archives upon completion."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(os.path.dirname(extract_out_path), exist_ok=True)
    fname = os.path.basename(out_path)
    fdir = os.path.dirname(out_path)
    # Ensure path exists
    if not os.path.isdir(fdir):
        raise ValueError(f'Path {fdir} does not exist, download aborted.')
    if progress:
        pbar = tqdm(total=byte_size, desc=f'Downloading file {fname}', unit='B', unit_scale=True)
    # Download file
    correct = False  # ensure file is not corrupted
    retries = RETRIES
    while not correct and retries > 0:  # Allow few failures
        session = requests.session()
        res = session.get(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) "
                                                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                                                      "Chrome/39.0.2171.95 "
                                                      "Safari/537.36"},
                          stream=True, verify=True)
        with open(out_path, 'wb') as fh:
            for chunk in res.iter_content(chunk_size=CHUNKSIZE):
                fh.write(chunk)
                if progress:
                    pbar.update(len(chunk))
                if callback is not None:
                    callback()
        correct = check_sha256sum(out_path, sha256sum) if sha256sum is not None else True
        if not correct:
            retries -= 1
            if progress:
                if retries > 0:
                    message = f'SHA256 hash unexpected for {fname}. Remaining download attempts: {retries}'
                else:
                    message = f'SHA256 hash unexpected for {fname}. All {RETRIES} attempts failed.'
                pbar.write(message)
            os.remove(out_path)
    if retries == 0:
        if progress:
            pbar.close()
        raise IOError(f'Download failed for {fname}')
    # Extract if ZIP file
    if fname.endswith('.zip'):
        with zipfile.ZipFile(out_path) as zip_handle:
            for name in zip_handle.namelist():
                subpath = extract_out_path if extract_out_path is not None else out_path
                zip_handle.extract(name, subpath)
        os.remove(out_path)
        if progress:
            pbar.close()
    # Extract if TAR.GZ file
    elif fname.endswith(('.tar.gz', '.tgz')):
        with tarfile.open(out_path) as tar_handle:
            tar_handle.extractall(extract_out_path
                                  if extract_out_path is not None
                                  else out_path)
        os.remove(out_path)
        if progress:
            pbar.close()


def check_sha256sum(filename, sha256):
    """Validate that the hex SHA256 digest of a target file matches expected checksum."""
    if not (isinstance(sha256, str) and len(sha256) == 64):
        raise ValueError("SHA256 must be 64 chars: {}".format(sha256))
    sha256_actual = sha256sum(filename)
    return sha256_actual == sha256


def sha256sum(filename, blocksize=None):
    """Compute the hexadecimal SHA256 hash of a file using buffered chunk streaming."""
    if blocksize is None:
        blocksize = 65536
    hash = hashlib.sha256()
    with open(filename, "rb") as fh:
        for block in iter(lambda: fh.read(blocksize), b""):
            hash.update(block)
    return hash.hexdigest()
