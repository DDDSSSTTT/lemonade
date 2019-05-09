""" 17 Category Flower Dataset

Credits: Maria-Elena Nilsback and Andrew Zisserman.
http://www.robots.ox.ac.uk/~vgg/data/flowers/17/

"""
from __future__ import absolute_import, print_function

import os
import sys
from six.moves import urllib
import tarfile

import numpy as np
import pickle

from data_utils import *
import file_fetch_ts as fetch

def load_data(dirname="./new_data_set/simple_train_set/NPYS/"):
    X=[]
    Y=[]
    speakers_list=fetch.get_spkrs(dirname)
    filenamelist=os.listdir(dirname)
    for filename in filenamelist:
        if filename.endswith(".npy"):
            load_spectrum=np.load(dirname+filename)
            unit_spec=np.reshape(load_spectrum,(80,800,1))
            X.append(unit_spec)
            speaker=fetch.extract(filename)
            Y.append(fetch.one_hot_from_item(speaker,speakers_list))
    X=np.array(X,dtype=np.float32)
    Y=np.array(Y,dtype=np.float64)
    return X, Y


def maybe_download(filename, source_url, work_directory):
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        print("Downloading Oxford 17 category Flower Dataset, Please "
              "wait...")
        filepath, _ = urllib.request.urlretrieve(source_url + filename,
                                                 filepath, reporthook)
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')

        untar(filepath, work_directory)
        build_class_directories(os.path.join(work_directory, 'jpg'))
    return filepath

#reporthook from stackoverflow #13881092
def reporthook(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize: # near the end
            sys.stderr.write("\n")
    else: # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))

def build_class_directories(dir):
    dir_id = 0
    class_dir = os.path.join(dir, str(dir_id))
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
    for i in range(1, 1361):
        fname = "image_" + ("%.4i" % i) + ".jpg"
        os.rename(os.path.join(dir, fname), os.path.join(class_dir, fname))
        if i % 80 == 0 and dir_id < 16:
            dir_id += 1
            class_dir = os.path.join(dir, str(dir_id))
            os.mkdir(class_dir)


def untar(fname, extract_dir):
    if fname.endswith("tar.gz") or fname.endswith("tgz"):
        tar = tarfile.open(fname)
        tar.extractall(extract_dir)
        tar.close()
        print("File Extracted")
    else:
        print("Not a tar.gz/tgz file: '%s '" % sys.argv[0])
