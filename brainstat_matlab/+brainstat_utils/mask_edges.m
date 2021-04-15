function [edges, idx] = mask_edges(edges, mask)
% MASK_EDGES    Remove edges with vertices outside the mask
%   edges = MASK_EDGES(edges, mask) removes edges connecting vertices outside
%   the mask. Returned edges have contiguous values. Edges is a e-by-2 matrix.
%   mask is a v-by-1 loigcal vector where v is the number of vertices. 
%
%   [edges, idx] = MASK_EDGES(edges, mask) also returns the row indices of the
%   edges that are kept.

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
