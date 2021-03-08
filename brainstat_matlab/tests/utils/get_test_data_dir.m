function test_data_dir = get_test_data_dir()
% Returns the path to the test data directory.

filepath = fileparts(mfilename('fullpath'));
brainstat_dir = fileparts(fileparts(filepath));

test_data_dir = strjoin({brainstat_dir, 'extern', 'test-data'}, filesep);
