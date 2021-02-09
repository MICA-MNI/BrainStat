function [ids,names] =  get_probes_from_genes(gene_names)
gene_list = sprintf('%s,',string(gene_names));
gene_list(end) = []; % Remove trailing comma. 

api_query = "?criteria=model::Probe," + ...
    "rma::criteria,[probe_type$eq'DNA'],products[abbreviation$eq'HumanMA'],gene[acronym$eq" + gene_list + "]" + ...
    ",rma::options[only$eq'probes.id','name']";
msg = genetics.api.query(api_query);

ids = [msg(:).id];
names = string({msg(:).name});
end
