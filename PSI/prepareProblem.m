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
S_init = S_init.seg;


%Make this a format that python can read
ground_truth.bw = ground_truth.seg.bw;
ground_truth.seg = ground_truth.seg.seg;
%ground_truth.M = reshape(M, size(M,1)*size(M,2), size(M,3));

ground_truth.Fu = ground_truth.unsuspected.Fu;
ground_truth.Su = ground_truth.unsuspected.Su;
ground_truth.unsuspectedPos = ground_truth.unsuspected.pos
ground_truth = rmfield(ground_truth, 'unsuspected');
% check_IM = ground_truth.IM(:);
% check_IM = check_IM + ground_truth.seg*(ground_truth.activity(:,1));
% check_IM = check_IM+ground_truth.Su*(1+ground_truth.Fu(:,1));
% %check that M(:,:,1) = check_im
% if abs(sum(sum(M(:,:,1)-reshape(check_IM, size(ground_truth.IM)))))>100
%     keyboard;
% end


save([basedir filesep problemname '.mat'], 'ground_truth', 'opts', 'obs', 'S_init');