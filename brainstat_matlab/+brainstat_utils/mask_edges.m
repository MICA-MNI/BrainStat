function [edges, idx] = mask_edges(edges, mask)
missing_edges = find(~mask);
remove_edges = ismember(edges, missing_edges);

idx = ~any(remove_edges,2);
edges = edges(idx,:); 
edges = make_contiguous(edges);
end

function Y = make_contiguous(Y)
val = unique(Y);
for ii = 1:numel(val)
    Y(Y==val(ii)) = ii;
end
end
