function [url, md5] = get_OSF_url(template, parcellation)
% GET_OSF_URL    gets URLs of datasets from OSF
%   [url, md5] = GET_OSF_URL(template) gets the url and md5 hash of the
%   surface template files of cortical surface files. Valid values for
%   template are 'fslr32k', 'fsaverage3', 'fsaverage4', 'fsaverage5',
%   'fsaverage6', 'fsaverage'.
%
%   [url, md5] = GET_OSF_URL(template, parcellation) returns the url of
%   'schaefer' or 'cammoun' parcellation files. 
%
%   OSF urls used here were retrieved from the netneurotools Python toolbox
%   published under the following license: 
%
% BSD 3-Clause License
% 
% Copyright (c) 2018, Network Neuroscience Lab All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
% 
% * Redistributions of source code must retain the above copyright notice,
% this
%   list of conditions and the following disclaimer.
% 
% * Redistributions in binary form must reproduce the above copyright
% notice,
%   this list of conditions and the following disclaimer in the
%   documentation and/or other materials provided with the distribution.
% 
% * Neither the name of the copyright holder nor the names of its
%   contributors may be used to endorse or promote products derived from
%   this software without specific prior written permission.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
% IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
% THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
% PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
% CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
% EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
% PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
% PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
% LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
% NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

arguments
    template (1,:) char 
    parcellation (1,:) char = ''
end

if strcmp(template, 'conte69')
    template = 'fslr32k';
end

json_file = [fileparts(mfilename('fullpath')), filesep, 'osf.json'];
fid = fopen(json_file); 
raw = fread(fid, inf); 
str = char(raw'); 
fclose(fid); 
json_contents = jsondecode(str);

if isempty(parcellation)
    switch template
        case 'fslr32k'
            json_field = json_contents.tpl_conte69;
        case {'fsaverage3', 'fsaverage4', 'fsaverage5', 'fsaverage6', 'fsaverage'}
            json_field = json_contents.tpl_fsaverage.(template);
        otherwise
            error('Unknown template %s.', template);
    end
else
    switch parcellation
        case {'cammoun', 'cammoun2012'}
            json_field = json_contents.atl_cammoun2012.(template);
        case {'schaefer', 'schaefer2018'}
            json_field = json_contents.atl_schaefer2018.(template);
        otherwise
            error('Unknown parcellation %s.', parcellation);
    end
end

url = sprintf('https://files.osf.io/v1/resources/%s/providers/osfstorage/%s', json_field.url{1}, json_field.url{2});
md5 = json_field.md5; 
end