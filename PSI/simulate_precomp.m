function simulate_precomp
startup
obs = []; opts = []; ground_truth = [];
load('PRECOMP_nonoise')

R = reconstruct_imaging_ADMM(obs,opts,10000, ground_truth);



keyboard