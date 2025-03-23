import math
import os
import shutil
import sys
import time
import torch


def load_vars(filepath):
    """
    Loads variables from the given filepath.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file '{filepath}' does not exist.")

    try:
        vs = torch.load(filepath)
        return vs
    except Exception as e:
        backup_filepath = '{}.old'.format(filepath)
        if os.path.exists(backup_filepath):
            shutil.copyfile(backup_filepath, filepath)
            os.remove(backup_filepath)
        raise e
    
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class DefaultList(list):
    def __getitem__(self, index):
        if index >= len(self):
            self.extend([Constants.deafaultlist_defaultValue] * (index - len(self) + 1))
            # self.extend([math.nan] * (index - len(self) + 1))
            
        return super().__getitem__(index)

# Classes
class Constants(object):
    eta = 1e-6
    log2 = math.log(2)
    log2pi = math.log(2 * math.pi)
    logceilc = 88  # largest cuda v s.t. exp(v) < inf
    logfloorc = -104  # smallest cuda v s.t. exp(v) > 0

    deafaultlist_defaultValue = 0
    OUTPUT_DIR = '../exps'

# https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
class Logger(object):
    def __init__(self, filename, mode="a"):
        self.terminal = sys.stdout
        self.log = open(filename, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.begin = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.begin
        self.elapsedH = time.gmtime(self.elapsed)
        print('====> [{}] Time: {:7.3f}s or {}'
              .format(self.name,
                      self.elapsed,
                      time.strftime("%H:%M:%S", self.elapsedH)))
