gii_surf_bs = read_gifti('/Users/reinder/GitHub/BrainStat_MICAMNI/shared/surfaces/conte69_32k_left_hemisphere.gii');
gii_surf_gf = gifti('/Users/reinder/GitHub/BrainStat_MICAMNI/shared/surfaces/conte69_32k_left_hemisphere.gii');
t(1) = all(all(gii_surf_bs.vertices == gii_surf_gf.vertices));
t(2) = all(all(gii_surf_bs.faces == gii_surf_gf.faces));
t(3) = all(all(gii_surf_bs.mat == gii_surf_gf.mat));

gii_label_bs = read_gifti('/Users/reinder/GitHub/micasoft/pipelines/08_micaProcessing/utilities/parcellations/economo_conte69_lh.label.gii');
gii_label_gf = gifti('/Users/reinder/GitHub/micasoft/pipelines/08_micaProcessing/utilities/parcellations/economo_conte69_lh.label.gii');
t(4) = all(all(gii_label_bs.cdata == gii_label_gf.cdata));
t(5) = all(all(gii_label_bs.labels.key == gii_label_gf.labels.key));
t(6) = all(all(gii_label_bs.labels.rgba == gii_label_gf.labels.rgba));
t(7) = all(all(gii_label_bs.labels.name == string(gii_label_gf.labels.name)));

gii_shape_bs = read_gifti('/Users/reinder/GitHub/micasoft/pipelines/08_micaProcessing/utilities/resample_fsaverage/fs_LR.L.midthickness_va_avg.59k_fs_LR.shape.gii');
gii_shape_gf = gifti('/Users/reinder/GitHub/micasoft/pipelines/08_micaProcessing/utilities/resample_fsaverage/fs_LR.L.midthickness_va_avg.59k_fs_LR.shape.gii');
t(8) = all(all(gii_shape_bs.cdata == gii_shape_gf.cdata));

gii_ascii_bs = read_gifti('/Users/reinder/Downloads/data_ASCII.gii');
gii_ascii_gf = gifti('/Users/reinder/Downloads/data_ASCII.gii');
t(9) = all(gii_ascii_gf.cdata == gii_ascii_bs.cdata);

gii_base64_bs = read_gifti('/Users/reinder/Downloads/data_BASE64_BINARY.gii');
gii_base64_gf = gifti('/Users/reinder/Downloads/data_BASE64_BINARY.gii');
t(10) = all(gii_base64_gf.cdata == gii_base64_bs.cdata);

gii_ext_bs = read_gifti('/Users/reinder/Downloads/data_EXTERNAL_FILD_BINARY.gii');
gii_ext_gf = gifti('/Users/reinder/Downloads/data_EXTERNAL_FILD_BINARY.gii');
t(11) = all((gii_ext_gf.cdata - gii_ext_bs.cdata)<1e-5); % Small inaccuracy due to double -> single conversion. 