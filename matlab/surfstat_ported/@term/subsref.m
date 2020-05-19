function r = subsref(obj,s)
if numel(s) > 1
    % If accessing the object in the "normal" manner
    r = obj;
    for ii = 1:numel(s)
        switch s(ii).type
            case '()'
                r = r(s(ii).subs{:});
            case '.'
                r = r.(s(ii).subs);
        end
    end
else
    switch s.type
        case '()'
            % User only requests with an index - grab the column of the
            % variance (same as with SurfStat term behavior).
            r = obj.matrix(s.subs{:});
        case '.'
            % User requests a particular field. Can either be a subfield or
            % name of a variable.
            f = properties(obj);
            if ismember(s.subs,f)
                r = obj.(s.subs);
            elseif ismember(s.subs,obj.names)
                r = obj.matrix(:,ismember(obj.names,s.subs));
            else
                error('Did not find the requested subfield and/or variable name.');               
            end
    end
end
end