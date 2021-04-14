import numpy as np
import pickle
from .testutil import datadir
from brainstat.stats.SLM import SLM
from brainstat.stats.terms import Term
from brainstat.stats._t_test import t_test
from brainstat.stats._multiple_comparisons import fdr


def generate_random_fdr_data(
    t_dim,
    df_max,
    finname,
    k=1,
    df_dim=None,
    mask_dim=None,
    dfs_max=None,
    tri_dim=None,
    resl_dim=None,
    du=None,
    c_dim=None,
    ef_dim=None,
    sd_dim=None,
    X_dim=None,
    coef_dim=None,
    SSE_dim=None,
    seed=0,
):
    """ Generate random test datasets. """
    # t_dim : tuple
    # df : int
    # k : int
    # finname : filename ending with *pkl
    np.random.seed(seed=seed)
    t = np.random.random_sample(t_dim)

    if df_dim is None:
        df = np.random.randint(0, high=df_max)

    D = {}
    D["t"] = t
    D["df"] = df
    D["k"] = k

    if mask_dim is not None:
        mask = np.random.choice(a=[False, True], size=mask_dim)
        D["mask"] = mask

    if dfs_max is not None:
        dfs = np.random.randint(1, dfs_max, size=t_dim)
        D["dfs"] = dfs

    if tri_dim is not None:
        tri = np.random.randint(1, np.size(t) - 1, size=tri_dim)
        D["tri"] = tri

    if resl_dim is not None:
        resl = np.random.random_sample(resl_dim)
        D["resl"] = resl

    if du is not None:
        D["du"] = du

    if c_dim is not None:
        c = np.random.random_sample(c_dim)
        D["c"] = c

    if ef_dim is not None:
        ef = np.random.random_sample(ef_dim)
        D["ef"] = ef

    if sd_dim is not None:
        sd = np.random.random_sample(sd_dim)
        D["sd"] = sd

    if X_dim is not None:
        first_col = np.ones(X_dim)
        seco_col = np.random.randint(10, 70, size=X_dim)
        X = np.concatenate((first_col, seco_col), axis=1)
        D["X"] = X

    if coef_dim is not None:
        coef = np.random.uniform(low=-2, high=2, size=coef_dim)
        D["coef"] = coef

    if SSE_dim is not None:
        SSE = np.random.uniform(low=0, high=3, size=SSE_dim)
        D["SSE"] = SSE

    with open(finname, "wb") as handle:
        pickle.dump(D, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return D


def get_fdr_output(D, foutname):
    """ Runs fdr and returns all relevant output. """

    slm = SLM(Term(1), Term(1))
    for key in D.keys():
        setattr(slm, key, D[key])

    # run fdr
    Q = fdr(slm)

    Q_out = {}
    Q_out["Q"] = Q

    with open(foutname, "wb") as handle:
        pickle.dump(Q_out, handle, protocol=pickle.HIGHEST_PROTOCOL)  #

    return


def generate_data_test_fdr():
    ### test_01 data in-out generation
    print("test_fdr: test_01 data is generated..")
    t_dim = (1, 64984)
    df_max = 64984
    finname = datadir("xstatq_01_IN.pkl")
    D = generate_random_fdr_data(t_dim, df_max, finname, seed=444)
    foutname = datadir("xstatq_01_OUT.pkl")
    get_fdr_output(D, foutname)

    ### test_02 data in-out generation
    print("test_fdr: test_02 data is generated..")
    t_dim = (1, 9850)
    df_max = 1000
    finname = datadir("xstatq_02_IN.pkl")
    D = generate_random_fdr_data(t_dim, df_max, finname, seed=445)
    foutname = datadir("xstatq_02_OUT.pkl")
    get_fdr_output(D, foutname)

    ### test_03 data in-out generation
    print("test_fdr: test_03 data is generated..")
    t_dim = (1, 2139)
    df_max = 2000
    k = 3
    finname = datadir("xstatq_03_IN.pkl")
    D = generate_random_fdr_data(t_dim, df_max, finname, k=k, seed=446)
    foutname = datadir("xstatq_03_OUT.pkl")
    get_fdr_output(D, foutname)

    ### test_04 data in-out generation
    print("test_fdr: test_04 data is generated..")
    t_dim = (1, 2475)
    df_max = 1500
    finname = datadir("xstatq_04_IN.pkl")
    mask_dim = 2475
    D = generate_random_fdr_data(t_dim, df_max, finname, mask_dim=mask_dim, seed=447)
    foutname = datadir("xstatq_04_OUT.pkl")
    get_fdr_output(D, foutname)

    ### test_05 data in-out generation
    print("test_fdr: test_05 data is generated..")
    t_dim = (1, 1998)
    df_max = 4000
    dfs_max = 1997
    finname = datadir("xstatq_05_IN.pkl")
    D = generate_random_fdr_data(t_dim, df_max, finname, dfs_max=dfs_max, seed=448)
    foutname = datadir("xstatq_05_OUT.pkl")
    get_fdr_output(D, foutname)

    ### test_06 data in-out generation
    print("test_fdr: test_06 data is generated..")
    t_dim = (1, 3328)
    df_max = 10000
    k = 2
    dfs_max = 3328
    mask_dim = 3328
    finname = datadir("xstatq_06_IN.pkl")
    D = generate_random_fdr_data(
        t_dim, df_max, finname, k=k, dfs_max=dfs_max, mask_dim=mask_dim, seed=449
    )
    foutname = datadir("xstatq_06_OUT.pkl")
    get_fdr_output(D, foutname)

    ### test_07 data in-out generation
    print("test_fdr: test_07 data is generated..")
    t_dim = (1, 9512)
    df_max = 5000
    dfs_max = 9511
    mask_dim = 9512
    tri_dim = (1724, 3)
    finname = datadir("xstatq_07_IN.pkl")
    D = generate_random_fdr_data(
        t_dim,
        df_max,
        finname,
        dfs_max=dfs_max,
        mask_dim=mask_dim,
        tri_dim=tri_dim,
        seed=450,
    )
    foutname = datadir("xstatq_07_OUT.pkl")
    get_fdr_output(D, foutname)

    ### test_08 data in-out generation
    print("test_fdr: test_08 data is generated..")
    t_dim = (1, 1520)
    df_max = 5000
    k = 5
    dfs_max = 9
    tri_dim = (4948, 3)
    resl_dim = (1520, 1)
    finname = datadir("xstatq_08_IN.pkl")
    D = generate_random_fdr_data(
        t_dim,
        df_max,
        finname,
        k=k,
        dfs_max=dfs_max,
        tri_dim=tri_dim,
        resl_dim=resl_dim,
        seed=451,
    )
    foutname = datadir("xstatq_08_OUT.pkl")
    get_fdr_output(D, foutname)

    ### test_09 data in-out generation
    print("test_fdr: test_09 data is generated..")
    t_dim = (1, 14397)
    df_max = 1
    dfs_max = 2
    tri_dim = (2734, 3)
    resl_dim = (8199, 1)
    # du = 9
    finname = datadir("xstatq_09_IN.pkl")
    D = generate_random_fdr_data(
        t_dim,
        df_max,
        finname,
        dfs_max=dfs_max,
        tri_dim=tri_dim,
        resl_dim=resl_dim,  # du = du,
        seed=452,
    )
    foutname = datadir("xstatq_09_OUT.pkl")
    get_fdr_output(D, foutname)

    ### test_10 data in-out generation
    print("test_fdr: test_10 data is generated..")
    t_dim = (1, 20484)
    df_max = 10
    tri_dim = (40960, 3)
    resl_dim = (61440, 1)
    c_dim = (1, 2)
    ef_dim = (1, 20484)
    sd_dim = (1, 20484)
    finname = datadir("xstatq_10_IN.pkl")
    D = generate_random_fdr_data(
        t_dim,
        df_max,
        finname,
        tri_dim=tri_dim,
        resl_dim=resl_dim,
        c_dim=c_dim,
        ef_dim=ef_dim,
        sd_dim=sd_dim,
        seed=453,
    )
    foutname = datadir("xstatq_10_OUT.pkl")
    get_fdr_output(D, foutname)

    ### test_11 data in-out generation
    print("test_fdr: test_11 data is generated..")
    t_dim = (1, 20484)
    df_max = 10
    tri_dim = (40960, 3)
    resl_dim = (61440, 1)
    c_dim = (1, 2)
    ef_dim = (1, 20484)
    sd_dim = (1, 20484)
    X_dim = (10, 2)
    coef_dim = (2, 20484)
    SSE_dim = (1, 20484)
    finname = datadir("xstatq_11_IN.pkl")
    D = generate_random_fdr_data(
        t_dim,
        df_max,
        finname,
        tri_dim=tri_dim,
        resl_dim=resl_dim,
        c_dim=c_dim,
        ef_dim=ef_dim,
        sd_dim=sd_dim,
        X_dim=X_dim,
        coef_dim=coef_dim,
        SSE_dim=SSE_dim,
        seed=454,
    )
    foutname = datadir("xstatq_11_OUT.pkl")
    get_fdr_output(D, foutname)

    ### test_12 data in-out generation
    print("test_fdr: test_12 data is generated..")
    t_dim = (1, 20484)
    df_max = 10
    tri_dim = (40960, 3)
    resl_dim = (61440, 1)
    c_dim = (1, 2)
    ef_dim = (1, 20484)
    sd_dim = (1, 20484)
    X_dim = (10, 2)
    coef_dim = (2, 20484)
    SSE_dim = (1, 20484)
    mask_dim = 20484
    finname = datadir("xstatq_12_IN.pkl")
    D = generate_random_fdr_data(
        t_dim,
        df_max,
        finname,
        tri_dim=tri_dim,
        mask_dim=mask_dim,
        resl_dim=resl_dim,
        c_dim=c_dim,
        ef_dim=ef_dim,
        sd_dim=sd_dim,
        X_dim=X_dim,
        coef_dim=coef_dim,
        SSE_dim=SSE_dim,
        seed=455,
    )
    foutname = datadir("xstatq_12_OUT.pkl")
    get_fdr_output(D, foutname)

    ### test_13 data in-out generation
    print("test_fdr: test_13 data is generated..")
    t_dim = (1, 20484)
    df_max = 10
    tri_dim = (40960, 3)
    resl_dim = (61440, 1)
    c_dim = (1, 2)
    ef_dim = (1, 20484)
    sd_dim = (1, 20484)
    mask_dim = 20484
    finname = datadir("xstatq_13_IN.pkl")
    D = generate_random_fdr_data(
        t_dim,
        df_max,
        finname,
        tri_dim=tri_dim,
        resl_dim=resl_dim,
        c_dim=c_dim,
        mask_dim=mask_dim,
        ef_dim=ef_dim,
        sd_dim=sd_dim,
        seed=453,
    )
    foutname = datadir("xstatq_13_OUT.pkl")
    get_fdr_output(D, foutname)

    #### test 14, real data
    realdataf = datadir("thickness_n10.pkl")
    ifile = open(realdataf, "br")
    DD = pickle.load(ifile)
    ifile.close()
    # run slm
    M = Term(DD["M"])
    slm = SLM(M, Term(1))
    slm.linear_model(DD["Y"])
    D = {}
    # run t-test
    t_test(slm)
    D["t"] = slm.t
    D["df"] = 10
    D["k"] = 1
    finname = datadir("xstatq_14_IN.pkl")
    with open(finname, "wb") as handle:
        pickle.dump(D, handle, protocol=pickle.HIGHEST_PROTOCOL)
    foutname = datadir("xstatq_14_OUT.pkl")
    get_fdr_output(D, foutname)
