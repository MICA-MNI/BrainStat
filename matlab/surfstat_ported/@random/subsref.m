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
            r = obj.variance(s.subs{:});
        case '.'
            % User just requests a particular field. 
            r = obj.(s.subs);
    end
end
end

