import numpy as np
import h5py
from .testutil import datadir
from brainstat.stats.SLM import SLM
from brainstat.stats.terms import Term


def generate_data_test_linear_model():
    ### test_01 data in-out generation
    print('test_linear_model: test_01 data is generated..')
    np.random.seed(seed=444)
    Y = np.random.rand(43,43)
    M = np.random.rand(43,43)
    M[:,0] = 1

    h = h5py.File(datadir('linmod_01_IN.h5'), 'w')
    h.create_dataset('Y', data=Y)
    h.create_dataset('M', data=M)
    h.close()

    # here we go
    slm = SLM(M, Term(1))
    slm.linear_model(Y)

    makeys = ['cluster_threshold', 'coef', 'df', 'drlim', 'model', 'niter',
              'SSE', 'thetalim', 'X']

    h = h5py.File(datadir('linmod_01_OUT.h5'), 'w')
    for makey in makeys:
        h.create_dataset(makey, data=getattr(slm, makey))
    h.close()


    ### test_02 data in-out generation
    print('test_linear_model: test_02 data is generated..')

    np.random.seed(seed=445)
    Y = np.random.rand(62,7)
    M = np.random.rand(62,92)
    M[:,0] = 1

    h = h5py.File(datadir('linmod_02_IN.h5'), 'w')
    h.create_dataset('Y', data=Y)
    h.create_dataset('M', data=M)
    h.close()

    slm = SLM(M, Term(1))
    slm.linear_model(Y)

    makeys = ['cluster_threshold', 'coef', 'df', 'drlim', 'model', 'niter',
              'SSE', 'thetalim', 'X']

    h = h5py.File(datadir('linmod_02_OUT.h5'), 'w')
    for makey in makeys:
        h.create_dataset(makey, data=getattr(slm, makey))
    h.close()


    ### test_03 data in-out generation
    print('test_linear_model: test_03 data is generated..')

    np.random.seed(seed=446)
    Y = np.random.rand(54,64,76)
    M = np.random.rand(54,2)
    M[:,0] = 1

    h = h5py.File(datadir('linmod_03_IN.h5'), 'w')
    h.create_dataset('Y', data=Y)
    h.create_dataset('M', data=M)
    h.close()

    slm = SLM(M, Term(1))
    slm.linear_model(Y)

    makeys = ['cluster_threshold', 'coef', 'df', 'drlim', 'model', 'niter',
              'SSE', 'thetalim', 'X']

    h = h5py.File(datadir('linmod_03_OUT.h5'), 'w')
    for makey in makeys:
        h.create_dataset(makey, data=getattr(slm, makey))
    h.close()


    ### test_04 data in-out generation
    print('test_linear_model: test_04 data is generated..')
    np.random.seed(seed=447)
    Y = np.random.rand(69,41,5)
    M = np.random.rand(69,30)
    M[:,0] = 1
    h = h5py.File(datadir('linmod_04_IN.h5'), 'w')
    h.create_dataset('Y', data=Y)
    h.create_dataset('M', data=M)
    h.close()

    slm = SLM(M, Term(1))
    slm.linear_model(Y)

    makeys = ['cluster_threshold', 'coef', 'df', 'drlim', 'model', 'niter',
              'SSE', 'thetalim', 'X']

    h = h5py.File(datadir('linmod_04_OUT.h5'), 'w')
    for makey in makeys:
        h.create_dataset(makey, data=getattr(slm, makey))
    h.close()


    ### test_05 data in-out generation
    print('test_linear_model: test_05 data is generated..')
    np.random.seed(seed=448)
    Y = np.random.rand(81,1)
    M = np.random.rand(81,2)
    M[:,0] = 1

    h = h5py.File(datadir('linmod_05_IN.h5'), 'w')
    h.create_dataset('Y', data=Y)
    h.create_dataset('M', data=M)
    h.close()

    slm = SLM(M, Term(1))
    slm.linear_model(Y)

    makeys = ['cluster_threshold', 'coef', 'df', 'drlim', 'model', 'niter',
              'SSE', 'thetalim', 'X']

    h = h5py.File(datadir('linmod_05_OUT.h5'), 'w')
    for makey in makeys:
        h.create_dataset(makey, data=getattr(slm, makey))
    h.close()


    ### test_06 data in-out generation
    print('test_linear_model: test_06 data is generated..')
    np.random.seed(seed=448)
    Y = np.random.rand(93,41,57)
    M = np.random.rand(93,67)
    M[:,0] = 1

    h = h5py.File(datadir('linmod_06_IN.h5'), 'w')
    h.create_dataset('Y', data=Y)
    h.create_dataset('M', data=M)
    h.close()

    slm = SLM(M, Term(1))
    slm.linear_model(Y)

    makeys = ['cluster_threshold', 'coef', 'df', 'drlim', 'model', 'niter',
              'SSE', 'thetalim', 'X']

    h = h5py.File(datadir('linmod_06_OUT.h5'), 'w')
    for makey in makeys:
        h.create_dataset(makey, data=getattr(slm, makey))
    h.close()


    ### test_07 data in-out generation
    print('test_linear_model: test_07 data is generated..')
    np.random.seed(seed=449)
    Y = np.random.rand(40,46,21)
    M = np.random.rand(40,81)
    M[:,0] = 1

    h = h5py.File(datadir('linmod_07_IN.h5'), 'w')
    h.create_dataset('Y', data=Y)
    h.create_dataset('M', data=M)
    h.close()

    slm = SLM(M, Term(1))
    slm.linear_model(Y)

    makeys = ['cluster_threshold', 'coef', 'df', 'drlim', 'model', 'niter',
              'SSE', 'thetalim', 'X']

    h = h5py.File(datadir('linmod_07_OUT.h5'), 'w')
    for makey in makeys:
        h.create_dataset(makey, data=getattr(slm, makey))
    h.close()


    ### test_08 data in-out generation
    print('test_linear_model: test_08 data is generated..')
    np.random.seed(seed=450)
    Y = np.random.rand(93,43)
    M = np.random.rand(93,2)
    M[:,0] = 1
    tri = np.random.randint(1,42,size=(93,3))

    h = h5py.File(datadir('linmod_08_IN.h5'), 'w')
    h.create_dataset('Y', data=Y)
    h.create_dataset('M', data=M)
    h.create_dataset('tri', data=tri)
    h.close()

    slm = SLM(M, Term(1))
    slm.surf = {'tri': tri}
    slm.linear_model(Y)

    makeys = ['cluster_threshold', 'coef', 'df', 'drlim', 'model', 'niter',
              'resl', 'SSE', 'thetalim', 'X', 'tri']

    h = h5py.File(datadir('linmod_08_OUT.h5'), 'w')
    for makey in makeys:
        h.create_dataset(makey, data=getattr(slm, makey))
    h.close()


    ### test_09 data in-out generation
    print('test_linear_model: test_09 data is generated..')
    np.random.seed(seed=451)
    Y = np.random.rand(98,69,60)
    M = np.random.rand(98,91)
    M[:,0] = 1
    tri = np.random.randint(1,68,size=(60,3))

    h = h5py.File(datadir('linmod_09_IN.h5'), 'w')
    h.create_dataset('Y', data=Y)
    h.create_dataset('M', data=M)
    h.create_dataset('tri', data=tri)
    h.close()

    slm = SLM(M, Term(1))
    slm.surf = {'tri': tri}
    slm.linear_model(Y)

    makeys = ['cluster_threshold', 'coef', 'df', 'drlim', 'model', 'niter',
              'resl', 'SSE', 'thetalim', 'X', 'tri']

    h = h5py.File(datadir('linmod_09_OUT.h5'), 'w')
    for makey in makeys:
        h.create_dataset(makey, data=getattr(slm, makey))
    h.close()


    ### test_10 data in-out generation
    print('test_linear_model: test_10 data is generated..')
    np.random.seed(seed=451)
    Y = np.random.rand(49,27)
    M = np.random.rand(49,2)
    M[:,0] = 1
    lat = np.random.randint(0,2,size=(3,3,3))

    h = h5py.File(datadir('linmod_10_IN.h5'), 'w')
    h.create_dataset('Y', data=Y)
    h.create_dataset('M', data=M)
    h.create_dataset('lat', data=lat)
    h.close()

    slm = SLM(M, Term(1))
    slm.surf = {'lat': lat}
    slm.linear_model(Y)

    makeys = ['cluster_threshold', 'coef', 'df', 'drlim', 'model', 'niter',
              'resl', 'SSE', 'thetalim', 'X', 'lat']

    h = h5py.File(datadir('linmod_10_OUT.h5'), 'w')
    for makey in makeys:
        h.create_dataset(makey, data=getattr(slm, makey))
    h.close()


    ### test_11 data in-out generation
    print('test_linear_model: test_11 data is generated..')
    np.random.seed(seed=451)
    Y = np.random.rand(45,27,3)
    M = np.random.rand(45, 7)
    M[:,0] = 1
    lat = np.random.randint(0,2,size=(3,3,3))

    h = h5py.File(datadir('linmod_11_IN.h5'), 'w')
    h.create_dataset('Y', data=Y)
    h.create_dataset('M', data=M)
    h.create_dataset('lat', data=lat)
    h.close()

    slm = SLM(M, Term(1))
    slm.surf = {'lat': lat}
    slm.linear_model(Y)

    makeys = ['cluster_threshold', 'coef', 'df', 'drlim', 'model', 'niter',
              'resl', 'SSE', 'thetalim', 'X', 'lat']

    h = h5py.File(datadir('linmod_11_OUT.h5'), 'w')
    for makey in makeys:
        h.create_dataset(makey, data=getattr(slm, makey))
    h.close()


    ### test_12 data: real in data
    print('test_linear_model: test_12 data is generated..')
    realdata = h5py.File(datadir('thickness_n10.h5'),'r')
    Y = np.array(realdata['Y'])
    M = np.array(realdata['AGE'])
    tri = np.array(realdata['tri'])

    AGE = Term(M, "AGE")
    M = 1 + AGE
    slm = SLM(M, Term(1))
    slm.surf = {'tri': tri} ########
    slm.linear_model(Y)

    makeys = ['cluster_threshold', 'coef', 'df', 'drlim', 'niter',
              'resl', 'SSE', 'thetalim', 'X', 'tri']

    h = h5py.File(datadir('linmod_12_OUT.h5'), 'w')
    for makey in makeys:
        h.create_dataset(makey, data=getattr(slm, makey))
    h.close()


    ### test_13: real in data shuffled
    print('test_linear_model: test_13 data is generated..')
    realdata = h5py.File(datadir('thickness_n10.h5'),'r')
    Y = np.array(realdata['Y'])
    np.random.seed(seed=452)
    np.random.shuffle(Y)
    M = np.array(realdata['AGE'])
    tri = np.array(realdata['tri'])

    h = h5py.File(datadir('linmod_13_IN.h5'), 'w')
    h.create_dataset('Y', data=Y)
    h.create_dataset('AGE', data=M)
    h.create_dataset('tri', data=tri)
    h.close()

    AGE = Term(M, "AGE")
    M = 1 + AGE
    slm = SLM(M, Term(1))
    slm.surf = {'tri': tri}
    slm.linear_model(Y)

    makeys = ['cluster_threshold', 'coef', 'df', 'drlim', 'niter',
              'resl', 'SSE', 'thetalim', 'X', 'tri']

    h = h5py.File(datadir('linmod_13_OUT.h5'), 'w')
    for makey in makeys:
        h.create_dataset(makey, data=getattr(slm, makey))
    h.close()


    ### test_14: real in data shuffled
    print('test_linear_model: test_14 data is generated..')
    realdata = h5py.File(datadir('thickness_n10.h5'),'r')
    Y = np.array(realdata['Y'])
    np.random.seed(seed=453)
    np.random.shuffle(Y)
    M = np.array(realdata['AGE'])
    np.random.seed(seed=454)
    tri = np.array(realdata['tri'])
    np.random.shuffle(tri)

    h = h5py.File(datadir('linmod_14_IN.h5'), 'w')
    h.create_dataset('Y', data=Y)
    h.create_dataset('AGE', data=M)
    h.create_dataset('tri', data=tri)
    h.close()

    AGE = Term(M, "AGE")
    M = 1 + AGE
    slm = SLM(M, Term(1))
    slm.surf = {'tri': tri}
    slm.linear_model(Y)

    makeys = ['cluster_threshold', 'coef', 'df', 'drlim', 'niter',
              'resl', 'SSE', 'thetalim', 'X', 'tri']

    h = h5py.File(datadir('linmod_14_OUT.h5'), 'w')
    for makey in makeys:
        h.create_dataset(makey, data=getattr(slm, makey))
    h.close()


    ### test_14: real in data shuffled
    print('test_linear_model: test_14 data is generated..')
    realdata = h5py.File(datadir('thickness_n10.h5'),'r')
    Y = np.array(realdata['Y'])
    np.random.seed(seed=453)
    np.random.shuffle(Y)
    M = np.array(realdata['AGE'])
    np.random.seed(seed=454)
    tri = np.array(realdata['tri'])
    np.random.shuffle(tri)

    h = h5py.File(datadir('linmod_14_IN.h5'), 'w')
    h.create_dataset('Y', data=Y)
    h.create_dataset('AGE', data=M)
    h.create_dataset('tri', data=tri)
    h.close()

    AGE = Term(M, "AGE")
    M = 1 + AGE
    slm = SLM(M, Term(1))
    slm.surf = {'tri': tri}
    slm.linear_model(Y)

    makeys = ['cluster_threshold', 'coef', 'df', 'drlim', 'niter',
              'resl', 'SSE', 'thetalim', 'X', 'tri']

    h = h5py.File(datadir('linmod_14_OUT.h5'), 'w')
    for makey in makeys:
        h.create_dataset(makey, data=getattr(slm, makey))
    h.close()


    ### test_15: real in data shuffled
    print('test_linear_model: test_15 data is generated..')
    realdata = h5py.File(datadir('thickness_n10.h5'),'r')
    Y = np.array(realdata['Y'])
    A = Y.copy()
    np.random.seed(seed=455)
    np.random.shuffle(Y)
    Y = np.concatenate((A, Y), axis=0)  # (20, 20484)
    a = np.ones((20,1))
    np.random.seed(seed=456)
    b = np.random.randint(22, 51, size=(20,1))
    np.random.seed(seed=457)
    c = np.random.randint(0,2, size=(20,1))
    np.random.seed(seed=458)
    d = np.random.randint(0,2, size=(20,1))
    e = np.zeros((20,1))
    f = np.ones((20,1))
    g = np.zeros((20,1))
    h = np.zeros((20,1))
    np.random.seed(seed=459)
    i = np.random.randint(10120, 22030, size=(20,1))
    M = np.concatenate((a,b,c,d,e,f,g,h,i), axis=1) # (20, 9)
    tri = np.array(realdata['tri']) # (40960, 3)

    h = h5py.File(datadir('linmod_15_IN.h5'), 'w')
    h.create_dataset('Y', data=Y)
    h.create_dataset('M', data=M)
    h.create_dataset('tri', data=tri)
    h.close()

    slm = SLM(M, Term(1))
    slm.surf = {'tri': tri}
    slm.linear_model(Y)

    makeys = ['cluster_threshold', 'coef', 'df', 'drlim', 'niter',
              'resl', 'SSE', 'thetalim', 'X', 'tri']

    h = h5py.File(datadir('linmod_15_OUT.h5'), 'w')
    for makey in makeys:
        h.create_dataset(makey, data=getattr(slm, makey))
    h.close()

