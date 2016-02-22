function prepareProblem(doInitialization)

if ~nargin
    doInitialization = false;
end

basedir = fileparts(which('prepareProblem'));

for problemNum = 5
switch problemNum
    case 1
        opts = default_opts;
        problemname = 'Problem_nonoise_1Kframes';
        opts.debug.nonoise = true;
        opts.sim.dur = 1000;
    case 2
        opts = default_opts;
        problemname = 'Problem_nonoise_20Kframes';
        opts.debug.nonoise = true;
        opts.sim.dur = 20000;
    case 3
        opts = default_opts;
        problemname = 'Problem_noise_1Kframes';
        opts.debug.nonoise = false;
        opts.sim.dur = 1000;
    case 4
        opts = default_opts;
        problemname = 'Problem_noise_20Kframes';
        opts.debug.nonoise = false;
        opts.sim.dur = 20000;
    case 5
        opts = default_opts;
        problemname = 'Problem_badInit'; %we will transpose the morphological image below
        opts.debug.nonoise = true;
        opts.sim.dur = 5000;
    case 6
        opts = default_opts;
        problemname = 'Problem_lowSNR';
        opts.debug.nonoise = false;
        opts.sim.dur = 1000;
        opts.scope.brightness = 0.05;
    case 7
        problemname = 'Problem_HPD_1Kframes';
        opts.scope.detectortype = 'HPD';
        opts = default_opts(opts);
        opts.debug.nonoise = false;
        opts.sim.dur = 1000;
    case 8
        opts = default_opts;
        problemname = 'Problem_perfectInit';
        opts.debug.nonoise = true;
        opts.sim.dur = 16;
        opts.sim.unsuspected.N = 0;
        %doInitialization = true;
end
opts.nframes = opts.sim.dur*opts.framerate;


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
[obs] = simulate_imaging(ground_truth, opts);
toc


%Motion correction
%motion = motion_SLAPmi(obs,opts);



%Segmentation for reconstruction; this has different parameters to reflect
%that this is unknown
if problemNum~=8 %for perfect initialization, use the same parameters
    opts.seg.dist_thresh = 0.75;  %in microns, the distance between seeds
    opts.seg.nh_size = 28;
    NSu = 5;
else
    NSu = 1;
end
disp('Segmenting image for reconstruction')

if problemNum==5 %Badinitialization
    S_init = segment_2D(obs.IM', opts);  %transpose image to get a really bad initialization
else
    S_init = segment_2D(obs.IM, opts);
end

S_bg = obs.IM; S_bg(S_init.bw) = 0;

Sk = S_init.seg./repmat(sum(S_init.seg,1), size(S_init.seg,1),1);
Su = rand(size(Sk,1), NSu);
Su(:,1) = S_bg(:);
Su = Su./repmat(sum(Su,1), size(Su,1),1);

if doInitialization
    S = [Sk Su];
    F_init = reconstruct_lightweight(obs,opts, S); %this takes ~3s/frame
    Fk = F_init(1:size(Sk,2),:);
    Fu = F_init(1+end-size(Su,2):end);
    
    %check how good the reconstruction is
%     M1 = ground_truth.seg.seg*ground_truth.activity(:,1) + ground_truth.IM(:);
%     M2 = S*F_init(:,1);
%     
%     %compare projection to measurement
%     obs1 = opts.P' * M1;
%     figure, plot(obs1-obs.data_in(:,1))
    
else
    Fk = [];
    Fu = [];
end


masks = Sk>eps;



%Make this a format that python can read
ground_truth.bw = ground_truth.seg.bw;
ground_truth.seg = ground_truth.seg.seg;
if isfield(ground_truth, 'unsuspected')
    ground_truth.Fu = ground_truth.unsuspected.Fu;
    ground_truth.Su = ground_truth.unsuspected.Su;
    ground_truth.unsuspectedPos = ground_truth.unsuspected.pos;
    ground_truth = rmfield(ground_truth, 'unsuspected');
else
    ground_truth.Fu = [];
    ground_truth.Su = [];
    ground_truth.unsuspectedPos = [];
end

% check_IM = ground_truth.IM(:);
% check_IM = check_IM + ground_truth.seg*(ground_truth.activity(:,1));
% check_IM = check_IM+ground_truth.Su*(1+ground_truth.Fu(:,1));
% %check that M(:,:,1) = check_im
% if abs(sum(sum(M(:,:,1)-reshape(check_IM, size(ground_truth.IM)))))>100
%     keyboard;
% end

save([basedir filesep problemname '.mat'], 'ground_truth', 'opts', 'obs', 'Sk', 'Fk', 'Su', 'Fu', 'masks');
end