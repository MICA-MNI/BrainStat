function home_dir = get_home_dir(type)
% GET_HOME_DIR    gets the user's home directory.

arguments
    type (1,:) char = 'char'
end

home_dir = java.lang.System.getProperty('user.home');
switch lower(type)
    case {'char', 'c'}
        home_dir = char(home_dir);
    case {'string', 's'}
        home_dir = string(home_dir);
    otherwise
        error('Unknown variable type.');
    end
end
end

