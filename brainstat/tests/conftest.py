# import git
from .datagen_linear_model import generate_data_test_linear_model


# This hook will run by pytest one time before execution all the particular.
# tests. It does checkout the needed 'extern/test-data' directory from the
# separate branch 'test-data'.
def pytest_configure(config):
    # repo = git.Repo(search_parent_directories=True)
    # origin = repo.remote()

    # cli = origin.repo.git
    # cli.checkout("origin/test-data", "--", "extern/test-data")
    # we have to reset the test-data, as checkout is staging it to the git index
    # cli.reset("--", "extern/test-data")

    ## let's generate the test data sets
    generate_data_test_linear_model()
