function home_dir = get_home_dir()
% GET_HOME_DIR    gets the user's home directory.
home_dir = char(java.lang.System.getProperty('user.home'));
end
