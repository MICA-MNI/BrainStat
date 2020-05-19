function [msg,meta] = query(command)

url = "http://api.brain-map.org/api/v2/data/query.json";
data = webread(url + command);

if data.success == 0
    disp(data.msg);
    error('Something failed in the API query. The previous message may provide a hint towards the cause.');
end

msg = data.msg; 

if isempty(msg)
    error('Server returned an empty message.')
end

if nargout > 1
    meta = rmfield(data,'msg');
end
end

