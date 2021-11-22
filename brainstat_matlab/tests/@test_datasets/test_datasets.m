classdef test_datasets < matlab.unittest.TestCase
    %TEST_DATASETS test class for testing the dataset functions.
    
    methods (Test)
        function test_fetch_template_surface(testCase)
            import matlab.unittest.constraints.HasField
            for surface = ["fsaverage" + (3:6), "fsaverage", "fslr32k"]
                if surface == "fslr32k"
                    layers = ["midthickness", "inflated", "vinflated"];
                else
                    layers = ["pial", "white", "sphere", "smoothwm", "inflated"];
                end
                for layer = layers
                    try
                        [surf_lh, surf_rh] = fetch_template_surface(surface, 'layer', layer{1});
                    catch err
                        if err.message == "Could not find the GIFTI library. Please install it from https://www.artefact.tk/software/matlab/gifti/";
                            % GIFTI library isn't installed on Github
                            % Actions
                            continue
                        else
                            rethrow(err)
                        end
                    end
                    for surf = {surf_lh, surf_rh}
                        testCase.verifyInstanceOf(surf{1}, 'struct');
                        testCase.verifyThat(surf{1}, HasField('faces'));
                        testCase.verifyThat(surf{1}, HasField('vertices'));
                    end
                end
            end
        end
        
        function test_fetch_parcellation(testCase)
            for atlas = ["cammoun", "schaefer", "glasser"]
                if atlas == "schaefer"
                    n_roi = [100, 200, 300, 400, 500, 600, 800, 1000];
                    templates = ["fsaverage5", "fsaverage6", "fsaverage", "fslr32k"];
                    seven_networks = [true, false];
                elseif atlas == "cammoun"
                    n_roi = [33, 60, 125, 250, 500];
                    seven_networks = true;
                    templates = ["fsaverage5", "fsaverage6", "fsaverage", "fslr32k"];
                elseif atlas == "glasser"
                    n_roi = 360;
                    seven_networks = true;
                    templates = ["fsaverage5", "fsaverage", "fslr32k"];
                end
                
                for n = n_roi
                    for template = templates
                        for networks = seven_networks
                            try
                                parcellation = fetch_parcellation(template, atlas, n, 'seven_networks', networks);
                            catch err
                                if err.identifier == "MATLAB:UndefinedFunction" && contains(err.message, '''gifti''')
                                    continue
                                else
                                    rethrow(err)
                                end
                            end
                            testCase.verifyEqual(double(min(parcellation)), 0);
                        end
                    end
                end
            end
        end
    end
end

