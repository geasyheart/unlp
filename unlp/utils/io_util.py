# -*- coding: utf8 -*-

#
import glob
import gzip
import json
import linecache
import os
import pickle
import platform
import random
import shutil
import tarfile
import urllib
import zipfile
from pathlib import Path
from typing import Tuple, Optional
from urllib.parse import urlparse

import requests

from unlp import pretrained
from unlp.utils.log_util import logger


def windows():
    system = platform.system()
    return system == 'Windows'


def unlp_home():
    if windows():
        return os.path.join(os.environ.get('APPDATA'), 'unlp')
    else:
        return os.path.join(os.path.expanduser("~"), '.unlp')


def get_resource(
        path: str, save_dir=unlp_home(),
        extract=True, prefix='https://file.hankcs.com/hanlp/',
        append_location=True,
):
    path = pretrained.ALL.get(path, path)
    anchor: str = None
    compressed = None
    if os.path.isdir(path):
        return path
    elif os.path.isfile(path):
        pass
    elif path.startswith('http:') or path.startswith('https:'):
        url = path
        if '#' in url:
            url, anchor = url.split('#', maxsplit=1)
        realpath = path_from_url(path, save_dir, prefix, append_location)
        realpath, compressed = split_if_compressed(realpath)
        # check if resource is there
        if anchor:
            if anchor.startswith('/'):
                # indicates the folder name has to be polished
                anchor = anchor.lstrip('/')
                parts = anchor.split('/')
                renamed_realpath = str(Path(realpath).parent.joinpath(parts[0]))
                if os.path.isfile(realpath + compressed):
                    os.rename(realpath + compressed, renamed_realpath + compressed)
                realpath = renamed_realpath
                anchor = '/'.join(parts[1:])
            child = path_join(realpath, anchor)
            if os.path.exists(child):
                return child
        elif os.path.isdir(realpath) or (os.path.isfile(realpath) and (compressed and extract)):
            return realpath
        else:
            if compressed:
                pattern = realpath + '.*'
                files = glob.glob(pattern)
                files = list(filter(lambda x: not x.endswith('.downloading'), files))
                zip_path = realpath + compressed
                if zip_path in files:
                    files.remove(zip_path)
                if files:
                    if len(files) > 1:
                        logger.debug(f'Found multiple files with {pattern}, will use the first one.')
                    return files[0]
        # realpath is where its path after exaction
        if compressed:
            realpath += compressed
        if not os.path.isfile(realpath):
            path = download(url=path, save_path=realpath)
        else:
            path = realpath
    if extract and compressed:
        path = uncompress(path)
        if anchor:
            path = path_join(path, anchor)

    return path


def split_file(filepath, train=0.8, dev=0.1, test=0.1, names=None, shuffle=False):
    num_samples = 0
    if filepath.endswith('.tsv'):
        for sent in read_tsv_as_sentence(filepath):
            num_samples += 1
    else:
        with open(filepath, encoding='utf-8') as src:
            for sample in src:
                num_samples += 1
    splits = {'train': train, 'dev': dev, 'test': test}
    splits = dict((k, v) for k, v in splits.items() if v)
    splits = dict((k, v / sum(splits.values())) for k, v in splits.items())
    accumulated = 0
    r = []
    for k, v in splits.items():
        r.append(accumulated)
        accumulated += v
        r.append(accumulated)
        splits[k] = accumulated
    if names is None:
        names = {}
    name, ext = os.path.splitext(filepath)
    filenames = [names.get(split, name + '.' + split + ext) for split in splits.keys()]
    outs = [open(f, 'w', encoding='utf-8') for f in filenames]
    if shuffle:
        shuffle = list(range(num_samples))
        random.shuffle(shuffle)
    if filepath.endswith('.tsv'):
        src = read_tsv_as_sentence(filepath)
    else:
        src = open(filepath, encoding='utf-8')
    for idx, sample in enumerate(src):
        if shuffle:
            idx = shuffle[idx]
        ratio = idx / num_samples
        for sid, out in enumerate(outs):
            if r[2 * sid] <= ratio < r[2 * sid + 1]:
                if isinstance(sample, list):
                    sample = '\n'.join('\t'.join(x) for x in sample) + '\n\n'
                out.write(sample)
                break
    if not filepath.endswith('.tsv'):
        src.close()
    for out in outs:
        out.close()
    return filenames


def path_from_url(url, save_dir=unlp_home(), prefix='https://file.hankcs.com/hanlp/', append_location=True):
    if not save_dir:
        save_dir = unlp_home()
    domain, relative_path = parse_url_path(url)
    if append_location:
        if not url.startswith(prefix):
            save_dir = os.path.join(save_dir, 'thirdparty', domain)
        else:
            # remove the relative path in prefix
            middle = prefix.split(domain)[-1].lstrip('/')
            if relative_path.startswith(middle):
                relative_path = relative_path[len(middle):]
        realpath = os.path.join(save_dir, relative_path)
    else:
        realpath = os.path.join(save_dir, os.path.basename(relative_path))
    return realpath


def parse_url_path(url):
    parsed: urllib.parse.ParseResult = urlparse(url)
    path = os.path.join(*parsed.path.strip('/').split('/'))
    return parsed.netloc, path


def path_join(path, *paths):
    return os.path.join(path, *paths)


def split_if_compressed(path: str, compressed_ext=('.zip', '.tgz', '.gz', 'bz2', '.xz')) -> Tuple[str, Optional[str]]:
    tar_gz = '.tar.gz'
    if path.endswith(tar_gz):
        root, ext = path[:-len(tar_gz)], tar_gz
    else:
        root, ext = os.path.splitext(path)
    if ext in compressed_ext or ext == tar_gz:
        return root, ext
    return path, None


def file_exist(filename) -> bool:
    return os.path.isfile(filename)


def remove_file(filename):
    if file_exist(filename):
        os.remove(filename)


def parent_dir(path):
    return os.path.normpath(os.path.join(path, os.pardir))


def download(url, save_path=None, save_dir=unlp_home(), prefix='https://file.hankcs.com/hanlp/', append_location=True):
    if not save_path:
        save_path = path_from_url(url, save_dir, prefix, append_location)
    if os.path.isfile(save_path):
        logger.info('Using local {}, ignore {}'.format(save_path, url))
        return save_path
    else:
        if not os.path.exists(parent_dir(save_path)):
            os.makedirs(parent_dir(save_path))
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    return save_path


def uncompress(path, dest=None, remove=True):
    """Uncompress a file and clean up uncompressed files once an error is triggered.

    Args:
      path: The path to a compressed file
      dest: The dest folder.
      remove: Remove archive file after decompression.
      verbose: ``True`` to print log message.

    Returns:
        Destination path.

    """
    # assert path.endswith('.zip')
    prefix, ext = split_if_compressed(path)
    folder_name = os.path.basename(prefix)
    file_is_zip = ext == '.zip'
    root_of_folder = None
    if ext == '.gz':
        try:
            with gzip.open(path, 'rb') as f_in, open(prefix, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        except Exception as e:
            remove_file(prefix)
            remove_file(path)
            raise e
    else:
        try:
            with zipfile.ZipFile(path, "r") if ext == '.zip' else tarfile.open(path, 'r:*') as archive:
                if not dest:
                    namelist = sorted(archive.namelist() if file_is_zip else archive.getnames())
                    if namelist[0] == '.':
                        namelist = namelist[1:]
                        namelist = [p[len('./'):] if p.startswith('./') else p for p in namelist]
                    if ext == '.tgz':
                        roots = set(x.split('/')[0] for x in namelist)
                        if len(roots) == 1:
                            root_of_folder = next(iter(roots))
                    else:
                        # only one file, root_of_folder = ''
                        root_of_folder = namelist[0].strip('/') if len(namelist) > 1 else ''
                    if all(f.split('/')[0] == root_of_folder for f in namelist[1:]) or not root_of_folder:
                        dest = os.path.dirname(path)  # only one folder, unzip to the same dir
                    else:
                        root_of_folder = None
                        dest = prefix  # assume zip contains more than one file or folder
                print('Extracting {} to {}'.format(path, dest))
                archive.extractall(dest)
                if root_of_folder:
                    if root_of_folder != folder_name:
                        # move root to match folder name
                        os.rename(path_join(dest, root_of_folder), path_join(dest, folder_name))
                    dest = path_join(dest, folder_name)
                elif len(namelist) == 1:
                    dest = path_join(dest, namelist[0])
        except Exception as e:
            remove_file(path)
            if os.path.exists(prefix):
                if os.path.isfile(prefix):
                    os.remove(prefix)
                elif os.path.isdir(prefix):
                    shutil.rmtree(prefix)
            raise e
    if remove:
        remove_file(path)
    return dest


def read_tsv_as_sentence(file_path, delimiter='\t'):
    sentence = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            cells = line.split(delimiter)
            if line and cells:
                sentence.append(cells)
            else:
                if sentence:
                    yield sentence
                    sentence = []

    if sentence:
        yield sentence


def get_lines(filename, idx, batch_size):
    start = (idx * batch_size)
    end = (idx + 1) * batch_size

    batch_lines = []
    for lineno in range(start, end):
        line = linecache.getline(filename, lineno=lineno).strip()
        if line:
            batch_lines.append(line)
    return batch_lines


def save_pickle(item, path):
    with open(path, 'wb') as f:
        pickle.dump(item, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_json(item: dict, path: str, ensure_ascii=False, cls=None, default=lambda o: repr(o), indent=2):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as out:
        json.dump(item, out, ensure_ascii=ensure_ascii, indent=indent, cls=cls, default=default)


def load_json(path):
    with open(path, encoding='utf-8') as src:
        return json.load(src)


def filename_is_json(filename):
    filename, file_extension = os.path.splitext(filename)
    return file_extension in ['.json', '.jsonl']


def merge_dict(d: dict, overwrite=False, inplace=False, **kwargs):
    nd = dict([(k, v) for k, v in d.items()] + [(k, v) for k, v in kwargs.items() if overwrite or k not in d])
    if inplace:
        d.update(nd)
        return d
    return nd
