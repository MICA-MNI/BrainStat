function varargout = read_surface_data(files)

if ischar(files)
    files = {files};
end

varargout = cell(1,numel(files));
for ii = 1:numel(files)
    if endsWith(files{ii}, {'.shape.gii','.func.gii'}) 
        gii = gifti(files{ii});
        varargout{ii} = gii.cdata;
    elseif endsWith(files{ii}, '.mat')
        obj = matfile(files{ii});
        f = fieldnames(obj);
        if numel(f) ~= 2
            error('Cannot load data from .mat files containing more than one variable.');
        end
        varargout{ii} = obj.(f{2});
    elseif endsWith(files{ii}, {'.txt','.thickness','.mgh','.asc'})
        varargout{ii} = SurfStatReadData1(files{ii});
    else
        error('Unknown file format.')
    end
end
end