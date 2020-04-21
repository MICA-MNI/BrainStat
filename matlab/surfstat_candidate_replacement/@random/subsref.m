function r = subsref(obj,s)
r = obj; 
if numel(s) > 1
    % If accessing the object in the "normal" manner
    for ii = 1:numel(s)
        switch s(ii).type
            case '()'
                r = r(s(ii).subs{:});
            case '.'
                r = r.(s(ii).subs);
        end
    end
else
    % If no field is provided, modify the object. 
    switch s.type
        case '()'
            r.data = r.data(s.subs{:});
        case '.'
            % User just requests a particular field. 
            r = r.(s.subs);
    end
end
end

