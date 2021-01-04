import os


def datadir(file):
    topdir = os.path.dirname(__file__)
    topdir = os.path.join(topdir, "../../../")
    topdir = os.path.abspath(topdir)
    return os.path.join(topdir, "extern/test-data", file)
