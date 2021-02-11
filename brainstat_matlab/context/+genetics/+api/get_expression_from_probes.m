function [expression, well_ids, donor_names] = get_expression_from_probes(probes)
command = "?criteria=service::human_microarray_expression[probes$eq" + sprintf('%0.f,', probes) + "]";
command = replace(command,',]',']');
msg = genetics.api.query(command);

for ii = 1:numel(probes)
    expression(:,ii) = cellfun(@str2double,msg.probes(ii).expression_level);
end

for ii = 1:numel(msg.samples)
    well_ids(ii) = msg.samples(ii).sample.well;
    donor_names{ii} = msg.samples(ii).donor.name;
end
end