function varargout = read_surface_data(files)

if ischar(files)
    files = {files};
end

varargout = cell(1,numel(files));
for ii = 1:numel(files)
    if endsWith(files{ii}, {'.shape.gii','.func.gii', '.label.gii'}) 
        gii = gifti(files{ii});
        varargout{ii} = gii.cdata;
    elseif endsWith(files{ii}, '.annot')
        varargout{ii} = annot2parcellation(files{ii});
    elseif endsWith(files{ii}, '.dlabel.nii')
        cii = io_utils.cifti_matlab.cifti_read(files{ii});
        varargout{ii} = cii.cdata;
    elseif endsWith(files{ii}, '.mat')
        obj = matfile(files{ii});
        f = fieldnames(obj);
        if numel(f) ~= 2
            error('Cannot load data from .mat files containing more than one variable.');
        end
        varargout{ii} = obj.(f{2});
    elseif endsWith(files{ii}, {'.txt','.thickness','.mgh','.asc'})
        varargout{ii} = io_utils.SurfStatReadData1(files{ii});
    else
        error('Unknown file format.')
    end
end
end

function parcellation = annot2parcellation(file)

try
[~, labels_tmp, color_table] = io_utils.freesurfer.read_annotation(file);
catch
    keyboard;
end
[vertex_id, labels_compress] = find(labels_tmp == color_table.table(:,5)');
[~, indices] = sort(vertex_id);
parcellation = labels_compress(indices);

% Sanity check that we find the correct number of labels:
if numel(parcellation) ~= numel(labels_tmp)
    error('Woops! Seems like something is wrong with this .annot file.');
end

end
