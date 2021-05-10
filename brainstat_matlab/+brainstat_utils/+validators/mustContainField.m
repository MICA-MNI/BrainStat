function mustContainField(x, fieldname)
% Validator function for checking whether a structure contains a field. 

if ~ismember(fieldname, fieldnames(x))
    eid = 'BrainStat:MissingField';
    msg = ['Structure must contain field: ''' fieldname '''.'];
    throwAsCaller(MException(eid, msg));
end
end
    
