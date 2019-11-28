# Module to download the dataset.

import os

import sys
if sys.version_info[0] > 2:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve
import hashlib
import zipfile
import tarfile

def require_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return None


def download(url, target_dir, filename=None):
    require_dir(target_dir)
    if filename is None:
        filename = url_filename(url)
    filepath = os.path.join(target_dir, filename)
    urlretrieve(url, filepath)
    return filepath


def url_filename(url):
    return url.split('/')[-1].split('#')[0].split('?')[0]


def check_sha256(file_name, orginal_sha256):
    # Open,close, read file and calculate SHA256 on its contents
    with open(file_name, 'rb') as f:
        hasher = hashlib.sha256()  # Make empty hasher to update piecemeal
        while True:
            block = f.read(64 * (
                1 << 20))  # Read 64 MB at a time; big, but not memory busting
            if not block:  # Reached EOF
                break
            hasher.update(block)  # Update with new block
    sha256_returned = hasher.hexdigest()
    # Finally compare original MD5 with freshly calculated
    if orginal_sha256 == sha256_returned:
        print('SHA256 verified.')
        return True
    else:
        print('SHA256 verification failed!')
        return False


def unzip(file, targetdir):
    with zipfile.ZipFile(file, "r") as zip_ref:
        zip_ref.extractall(targetdir)

def untar(file, targetdir):
    with tarfile.open(file) as tf:
        tf.extractall(targetdir)

if __name__ == '__main__':
    
    url_maestro = 'https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0.zip'
    sha256_maestro = '572c6054e8d2c7219aa4df9a29357da0f9789524c11fa38cef7d4bd8542c93f0'

    print('Download Maestro')
    download(url_maestro, '../data/maestro')
    assert(check_sha256('../data/maestro/maestro-v2.0.0.zip', sha256_maestro))
    print('Extract Maestro dataset')
    unzip('../data/maestro/maestro-v2.0.0.zip', '../data/')
    os.remove('../data/maestro/maestro-v2.0.0.zip')