import os

def make_dir(path_to_save):
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
