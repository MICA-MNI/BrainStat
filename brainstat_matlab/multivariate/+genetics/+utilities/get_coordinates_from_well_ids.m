function mni_coordinates = get_coordinates_from_well_ids(well_ids,voxel_size)

if numel(unique(well_ids)) ~= numel(well_ids)
    error('Found non-unique IDs in the input.')
end

% Read the wells CSV file. 
path = string(fileparts(mfilename('fullpath')));
wells = csvread(path + filesep + "corrected_mni_coordinates.csv", 1, 0);

% Find rows with requested well IDs.
[I,~] = find(wells(:,1) == well_ids(:)'); 
if numel(I) ~= numel(well_ids)
    error('Could not find a well for every well ID.')
end

% Return coordinates of requested well IDs.
mni_coordinates = wells(I,2:end);
end