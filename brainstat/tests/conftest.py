"""Configurations for pytest."""
import git


# This hook will run by pytest one time before execution all the particular.
# tests. It does checkout the needed 'extern/test-data' directory from the
# separate branch 'test-data'.
def pytest_configure(config):
    repo = git.Repo(search_parent_directories=True)
    origin = repo.remote()

    cli = origin.repo.git
    cli.checkout("origin/test-data-2.0", "--", "extern/test-data")
    # we have to reset the test-data, as checkout is staging it to the git index
    cli.reset("--", "extern/test-data")
