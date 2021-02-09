function S2 = convert_surface(S,varargin)
% CONVERT_SURFACE   Converts surfaces to MATLAB or SurfStat format and 
%                   writes new surface files.
%
%   S2 = CONVERT_SURFACE(S) converts surface S to SurfStat format. S can
%   either be a file (.gii, .mat, .obj, Freesurfer), a loaded variable (in
%   SurfStat or MATLAB format), or a cell array containing multiple of the
%   former.
%
%   S2 = CONVERT_SURFACE(S,'format',F) allows for specifying the output
%   format F, either 'SurfStat' or 'MATLAB'.
%
%   S2 = CONVERT_SURFACE(S,'path',P) will write a file to path P. Supported
%   formats are .gii, .obj, .mat and Freesurfer. Only one surface can be
%   provided in S when writing surfaces. 
%
%   For more information, pleasse consult our <a
% href="https://brainspace.readthedocs.io/en/latest/pages/matlab_doc/support_functions/convert_surface.html">ReadTheDocs</a>.

% Read function input.
p = inputParser;
check_input = @(x) iscell(x) || ischar(x);
addParameter(p,'format','SurfStat', check_input);
addParameter(p,'path','', @ischar)
parse(p, varargin{:});

format = lower(p.Results.format);
path = p.Results.path;

% If multiple surfaces provided in a cell, loop over all recursively.
if iscell(S)
    if ~isempty(path) && numel(S) > 1
        error('Multiple inputs are not supported for surface writing.');
    end
    for ii = 1:numel(S)
        S2{ii} = convert_surface(S{ii},'format',format);
    end
    return
end

if ischar(S)
    % If input is a path rather than a surface in memory, read it. 
    S = local_surface_reader(S);
end
[faces, vertices] = local_common_format(S); 
S2 = local_surface_conversion(faces, vertices, format); 

if ~isempty(path)
    local_surface_writer(S2, path);
end
end

function S = local_surface_reader(S)
% Reads input surfaces. Support formats are GIFTI, .mat, .obj, and
% FreeSurfer formats. Note that no format checks are performed on .mat
% files. 

% If input is a char array, load the file. 
if endsWith(S,'.gii')
    if ~exist('gifti.m','file')
        error('Could not find the GIFTI library. Please install it from https://www.artefact.tk/software/matlab/gifti/');
    end
    S = gifti(S);
elseif endsWith(S, '.mat')
    S = load(S);
else
    try
        S = SurfStatReadSurf1(S);
    catch
        error('Could not read surface.');
    end
end

end

function [faces, vertices] = local_common_format(S)
% Converts the input surfaces to a common faces/vertices format (i.e.
% standard MATLAB format). Currently supprts SurfStat GIFTI, SurfStat, and
% MATLAB formats. 

f = fieldnames(S);

if ismember('tri',f) && ismember('coord',f) && ismember('faces',f) && ismember('vertices',f)
    % Sanity check
    error('Could not determine input surface type.');
elseif ismember('faces',f) && ismember('vertices',f)
    % GIFTI/MATLAB format.
    faces = S.faces;
    vertices = S.vertices;
elseif ismember('tri',f) && ismember('coord',f)
    % SurfStat format. 
    faces = S.tri;
    vertices = S.coord';
else
    error('Could not determine input surface type.');
end
end

function S2 = local_surface_conversion(faces, vertices, format)
% Converts input surface described by variabels faces and vertices to an
% output format. Valid formats are 'surfstat' and 'matlab'.

switch format
    case 'surfstat'
        S2.tri = faces;
        S2.coord = vertices';
    case 'matlab'
        S2.faces = faces;
        S2.vertices = vertices;
    otherwise
        error('Unknown output type requested. Options are: ''surfstat'' and ''matlab''.');
end
end

function local_surface_writer(faces, vertices, path)
% Writes a surface described by variables faces and vertices to a file with
% name path. Output format is determined from the file extension. Supported
% file extensions are .gii, .mat, and .obj. If another file extension is
% provided, then this functions defaults to Freesurfer format. 

if endsWith(path,'.gii')
    % Write gifti file.
    if ~exist('gifti.m','file')
        error('Could not find the GIFTI library. Please install it from https://www.artefact.tk/software/matlab/gifti/');
    end
    S = struct('faces', faces, 'vertices', vertices);
    gii = gifti(S); 
    save(gii,path);
elseif endsWith(path,'.mat')
    % Write matlab file.
    switch lower(format)
        case 'surfstat'
            tri = faces; 
            coord = vertices';
            save(path,'tri','coord');
        case 'matlab'
            save(path,'faces','vertices');
    end
else
    % Assume .obj or Freesurfer.
    if ~endsWith(path, '.obj')
        warning('Did not recognize the file extension. Saving as a Freesurfer file.');
    end
    S.tri = faces; 
    S.coord = vertices';
    SurfStatWriteSurf1(path,S);    
end

end