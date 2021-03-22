import numpy as np
import pickle
from .testutil import datadir
from brainstat.stats.SLM import SLM
from brainstat.stats.terms import Term


def generate_random_test_data(Y_dim, M_dim, finname, seed=0,
                              triD = None, latD = None,
                              M_term = False, add_intercept=True):
    """ Generate random test datasets. """
    # Y_dim : tuple
    # M_dim : tuple
    # finname : filename ending with *pkl
    np.random.seed(seed=seed)
    Y = np.random.random_sample(Y_dim)
    M = np.random.random_sample(M_dim)
    if add_intercept:
        M = np.concatenate((np.ones((M_dim[0],1)), M), axis=1)
    if M_term:
        M = Term(M)

    D = {}
    D['Y'] = Y
    D['M'] = M

    if triD is not None:
        tri = np.random.randint(triD['tri_min'], triD['tri_max'],
                                size=triD['tri_dim'])
        D['tri'] = tri

    if latD is not None:
        lat = np.random.randint(latD['lat_min'], latD['lat_max'],
                                size=latD['lat_dim'])
        D['lat'] = lat

    with open(finname, 'wb') as handle:
        pickle.dump(D, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if triD is not None:
        return Y, M, tri
    elif latD is not None:
        return Y, M, lat
    else:
        return Y, M


def get_linmod_output(Y, M, foutname, tri=None, lat=None):
    """ Runs linmod and returns all relevant output. """
    slm = SLM(M, Term(1))

    if tri is not None:
        slm.surf = {'tri': tri}
    if lat is not None:
        slm.lat = {'lat': lat}

    slm.linear_model(Y)

    keys = [
        "cluster_threshold",
        "coef",
        "df",
        "drlim",
        "niter",
        "resl",
        "SSE",
        "thetalim",
        "X",
        "tri",
    ]

    D = {}
    for key in keys:
        if getattr(slm, key) is not None:
            D[key] = getattr(slm, key)

    with open(foutname, 'wb') as handle:
        pickle.dump(D, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return D


def generate_data_test_linear_model():
    ### test_01 data in-out generation
    print("test_linear_model: test_01 data is generated..")
    Y_dim = (43, 43)
    M_dim = (43, 43)
    finname = datadir("linmod_01_IN.pkl")
    Y, M = generate_random_test_data(Y_dim, M_dim, finname, seed=444)
    foutname = datadir("linmod_01_OUT.pkl")
    get_linmod_output(Y, M, foutname)


    ### test_02 data in-out generation
    print("test_linear_model: test_02 data is generated..")
    Y_dim = (62, 7)
    M_dim = (62, 92)
    finname = datadir("linmod_02_IN.pkl")
    Y, M = generate_random_test_data(Y_dim, M_dim, finname, seed=445)
    foutname = datadir("linmod_02_OUT.pkl")
    get_linmod_output(Y, M, foutname)


    ### test_03 data in-out generation
    print("test_linear_model: test_03 data is generated..")
    Y_dim = (54, 64, 76)
    M_dim = (54, 2)
    finname = datadir("linmod_03_IN.pkl")
    Y, M = generate_random_test_data(Y_dim, M_dim, finname, seed=446)
    foutname = datadir("linmod_03_OUT.pkl")
    get_linmod_output(Y, M, foutname)


    ### test_04 data in-out generation
    print("test_linear_model: test_04 data is generated..")
    Y_dim = (69, 41, 5)
    M_dim = (69, 30)
    finname = datadir("linmod_04_IN.pkl")
    Y, M = generate_random_test_data(Y_dim, M_dim, finname, seed=447)
    foutname = datadir("linmod_04_OUT.pkl")
    get_linmod_output(Y, M, foutname)


    ### test_05 data in-out generation
    print("test_linear_model: test_05 data is generated..")
    Y_dim = (81, 1)
    M_dim = (81, 2)
    finname = datadir("linmod_05_IN.pkl")
    Y, M = generate_random_test_data(Y_dim, M_dim, finname, seed=448)
    foutname = datadir("linmod_05_OUT.pkl")
    get_linmod_output(Y, M, foutname)


    ### test_06 data in-out generation
    print("test_linear_model: test_06 data is generated..")
    Y_dim = (93, 41, 57)
    M_dim = (93, 67)
    finname = datadir("linmod_06_IN.pkl")
    Y, M = generate_random_test_data(Y_dim, M_dim, finname, seed=448)
    foutname = datadir("linmod_06_OUT.pkl")
    get_linmod_output(Y, M, foutname)


    ### test_07 data in-out generation
    print("test_linear_model: test_07 data is generated..")
    Y_dim = (40, 46, 21)
    M_dim = (40, 81)
    finname = datadir("linmod_07_IN.pkl")
    Y, M = generate_random_test_data(Y_dim, M_dim, finname, seed=449)
    foutname = datadir("linmod_07_OUT.pkl")
    get_linmod_output(Y, M, foutname)


    ### test_08 data in-out generation
    print("test_linear_model: test_08 data is generated..")
    Y_dim = (93, 43)
    M_dim = (93, 2)
    triD = {}
    triD['tri_min'] = 1
    triD['tri_max'] = 42
    triD['tri_dim'] = (93, 3)
    finname = datadir("linmod_08_IN.pkl")
    Y, M, tri = generate_random_test_data(Y_dim, M_dim, finname, seed=450,
                                          triD=triD)
    foutname = datadir("linmod_08_OUT.pkl")
    get_linmod_output(Y, M, foutname, tri=tri)


    ### test_09 data in-out generation
    print("test_linear_model: test_09 data is generated..")
    Y_dim = (98, 69, 60)
    M_dim = (98, 91)
    triD = {}
    triD['tri_min'] = 1
    triD['tri_max'] = 68
    triD['tri_dim'] = (60, 3)
    finname = datadir("linmod_09_IN.pkl")
    Y, M, tri = generate_random_test_data(Y_dim, M_dim, finname, seed=451,
                                          triD=triD)
    foutname = datadir("linmod_09_OUT.pkl")
    get_linmod_output(Y, M, foutname, tri=tri)


    ### test_10 data in-out generation
    print("test_linear_model: test_10 data is generated..")
    Y_dim = (49, 27)
    M_dim = (49, 2)
    latD = {}
    latD['lat_min'] = 0
    latD['lat_max'] = 2
    latD['lat_dim'] = (3, 3, 3)
    finname = datadir("linmod_10_IN.pkl")
    Y, M, lat = generate_random_test_data(Y_dim, M_dim, finname, seed=452,
                                          latD=latD)
    foutname = datadir("linmod_10_OUT.pkl")
    get_linmod_output(Y, M, foutname, lat=lat)


    ### test_11 data in-out generation
    print("test_linear_model: test_11 data is generated..")
    Y_dim = (45, 27, 3)
    M_dim = (45, 7)
    latD = {}
    latD['lat_min'] = 0
    latD['lat_max'] = 2
    latD['lat_dim'] = (3, 3, 3)
    finname = datadir("linmod_11_IN.pkl")
    Y, M, lat = generate_random_test_data(Y_dim, M_dim, finname, seed=453,
                                          latD=latD)
    foutname = datadir("linmod_11_OUT.pkl")
    get_linmod_output(Y, M, foutname, lat=lat)


    ### test_12 data in-out generation
    print("test_linear_model: test_12 data is generated..")
    # this is real data, save manually..'
    realdataf = datadir("thickness_n10.pkl")
    ifile = open(realdataf, "br")
    D = pickle.load(ifile)
    ifile.close()
    Y = D['Y']
    M = D['M']
    tri = D['tri']
    foutname = datadir("linmod_12_OUT.pkl")
    get_linmod_output(Y, M, foutname, tri=tri)


    ### test_13: real in data shuffled
    print("test_linear_model: test_13 data is generated..")
    realdataf = datadir("thickness_n10.pkl")
    ifile = open(realdataf, "br")
    D = pickle.load(ifile)
    ifile.close()
    Y = D['Y']
    np.random.seed(seed=454)
    np.random.shuffle(Y)
    M = D['M']
    tri = D['tri']
    finname = datadir("linmod_13_IN.pkl")
    with open(finname, 'wb') as handle:
        pickle.dump(D, handle, protocol=pickle.HIGHEST_PROTOCOL)
    foutname = datadir("linmod_13_OUT.pkl")
    get_linmod_output(Y, M, foutname, tri=tri)


    ### test_14: real in data shuffled
    print("test_linear_model: test_14 data is generated..")
    realdataf = datadir("thickness_n10.pkl")
    ifile = open(realdataf, "br")
    Din = pickle.load(ifile)
    ifile.close()
    Y = Din['Y']
    M = Din['M']
    tri = Din['tri']
    np.random.seed(seed=455)
    np.random.shuffle(Y)
    np.random.seed(seed=456)
    np.random.shuffle(tri)
    # save
    D = {}
    D['Y'] = Y
    D['M'] = M
    D['tri'] = tri
    finname = datadir("linmod_14_IN.pkl")
    with open(finname, 'wb') as handle:
        pickle.dump(D, handle, protocol=pickle.HIGHEST_PROTOCOL)
    foutname = datadir("linmod_14_OUT.pkl")
    get_linmod_output(Y, M, foutname, tri=tri)


    ### test_15: real in data shuffled and is manually extended
    print("test_linear_model: test_15 data is generated..")
    realdataf = datadir("thickness_n10.pkl")
    ifile = open(realdataf, "br")
    Din = pickle.load(ifile)
    ifile.close()
    Y = Din['Y']
    A = Y.copy()
    # extend Y
    np.random.seed(seed=457)
    np.random.shuffle(Y)
    Y = np.concatenate((A, Y), axis=0)  # (20, 20484)
    # generate M manually
    a = np.ones((20, 1))
    np.random.seed(seed=456)
    b = np.random.randint(22, 51, size=(20, 1))
    np.random.seed(seed=457)
    c = np.random.randint(0, 2, size=(20, 1))
    np.random.seed(seed=458)
    d = np.random.randint(0, 2, size=(20, 1))
    e = np.zeros((20, 1))
    f = np.ones((20, 1))
    g = np.zeros((20, 1))
    h = np.zeros((20, 1))
    np.random.seed(seed=459)
    i = np.random.randint(10120, 22030, size=(20, 1))
    M = np.concatenate((a,b,c,d,e,f,g,h,i), axis=1) # (20, 9)
    # get tri from real data
    tri = Din['tri'] # (40960, 3)
    # save
    D = {}
    D['Y'] = Y
    D['M'] = M
    D['tri']= tri
    finname = datadir("linmod_15_IN.pkl")
    with open(finname, 'wb') as handle:
        pickle.dump(D, handle, protocol=pickle.HIGHEST_PROTOCOL)
    foutname = datadir("linmod_15_OUT.pkl")
    get_linmod_output(Y, M, foutname, tri=tri)

