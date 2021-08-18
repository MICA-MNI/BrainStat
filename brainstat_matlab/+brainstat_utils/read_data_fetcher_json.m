function json = read_data_fetcher_json()
% Reads the .json file containing URLs of external data. 
json_path = brainstat_utils.get_brainstat_directories('brainstat_precomputed_data') + filesep + "data_urls.json";
fid = fopen(json_path); 
raw = fread(fid, inf); 
str = char(raw'); 
fclose(fid); 
json = jsondecode(str);
end