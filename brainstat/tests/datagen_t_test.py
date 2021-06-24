import numpy as np
from brainstat.tests.testutil import _generate_sphere, generate_random_data_model, array2effect, slm2files, save_input_dict, datadir
import pickle
from brainstat.stats.SLM import SLM
from sklearn.model_selection import ParameterGrid
from brainspace.mesh.mesh_elements import get_points


running_as_Peer = True

def generate_test_data():
    np.random.seed(0)
    surface = _generate_sphere()
    parameters = [
        {
            "n_observations": [103],
            "n_vertices": [np.array(get_points(surface)).shape[0]],
            "n_variates": [1, 2, 3],
            "n_random": [0],
            "n_predictors": [1, 7],
            "surf": [None, surface],
        },
        {
            "n_observations": [103],
            "n_vertices": [np.array(get_points(surface)).shape[0]],
            "n_variates": [1],
            "n_random": [1],
            "n_predictors": [2, 7],
            "surf": [None, surface],
        },
    ]
    
    test_num = 0
    for params in ParameterGrid(parameters):
        test_num += 1
        Y, M = generate_random_data_model(params["n_observations"], params["n_vertices"], params["n_variates"], params["n_predictors"])
        model = array2effect(M, params["n_random"])
        contrast = -M[:, -1]

        # Run once before saving data.
        slm_dumbdumb = SLM(model, contrast, params["surf"])
        slm_dumbdumb.linear_model(Y)
        slm_dumbdumb.t_test()

        save_input_dict(
            {"Y": Y, "M": M, "contrast": contrast, "surf": params["surf"], "n_random": params["n_random"]},
            "xstatt",
            test_num,
        )

        # Load the file we just saved and rerun.
        infile = datadir("xstatt_" + f"{test_num:02d}" + "_IN.pkl")
        ifile = open(infile, "br")
        idic = pickle.load(ifile)
        ifile.close()
        model = array2effect(M, params["n_random"])
        
        slm = SLM(model, contrast, params["surf"])
        slm.linear_model(idic["Y"])
        slm.t_test()
        slm2files(slm, "xstatt", test_num)

        if running_as_Peer and test_num == 13:
            import pdb
            pdb.set_trace()
            # slm.SSE - slm_dumbdumb.SSE


if __name__ == "__main__":
    generate_test_data()
