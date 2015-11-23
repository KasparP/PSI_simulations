function prepareProblem

opts = default_opts;

basedir = which('prepareProblem');
problemname = 'Problem_nonoise_v1';

%Begin Simulation
disp('Generating ground truth.')
tic
ground_truth = simulate_sample(opts);
toc

%%
disp('Generating projections.')
tic
[opts.P, opts.R]= generate_projections(size(ground_truth.IM), opts);
toc
%%

disp('Simulating imaging.')
tic
[obs, M] = simulate_imaging(ground_truth, opts);
toc



opts.seg.dist_thresh = 0.65;  %in microns, the distance between seeds
disp('Segmenting image for reconstruction')
S_init = segment_2D(obs.IM, opts);



save([problemname '.mat'], 'ground_truth', 'opts', 'obs', 'S_init');
keyboard