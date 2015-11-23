function prepareProblem

opts = default_opts;

basedir = fileparts(which('prepareProblem'))
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



%change some of the segmentation parameters
opts.seg.dist_thresh = 0.75;  %in microns, the distance between seeds
opts.seg.nh_size = 28;
disp('Segmenting image for reconstruction')
S_init = segment_2D(obs.IM, opts);



save([basedir filesep problemname '.mat'], 'ground_truth', 'opts', 'obs', 'S_init');
keyboard