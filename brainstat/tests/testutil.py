import os


def datadir(file):
    mydir = os.path.dirname(__file__)
    return os.path.join(mydir, "data", file)
