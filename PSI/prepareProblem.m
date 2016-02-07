function prepareProblem(doInitialization)

if ~nargin
    doInitialization = false;
end

opts = default_opts;

basedir = fileparts(which('prepareProblem'));
problemname = 'Problem_nonoise_v2';

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


%Motion correction
%motion = motion_SLAPmi(obs,opts);

%Segmentation for reconstruction; this has different parameters to reflect
%that this is unknown
opts.seg.dist_thresh = 0.75;  %in microns, the distance between seeds
opts.seg.nh_size = 28;
disp('Segmenting image for reconstruction')
S_init = segment_2D(obs.IM, opts);

S_bg = obs.IM; S_bg(S_init.bw) = 0;

NSu = 5;
Sk = S_init.seg./repmat(sum(S_init.seg,1), size(S_init.seg,1),1);
Su = rand(size(Sk,1), NSu);
Su(:,1) = S_bg(:);
Su = Su./repmat(sum(Su,1), size(Su,1),1);

if doInitialization
    S = [Sk Su];
    F_init = reconstruct_lightweight(obs,opts, S); %this takes ~3s/frame
    Fk = F_init(1:size(Sk,2),:);
    Fu = F_init(1+end-size(Su,2):end);
else
    Fk = [];
    Fu = [];
end


masks = Sk>eps;
Sk_masked = cell(size(Sk, 2),1);
for ix = 1:length(Sk_masked)
    Sk_masked{ix} = Sk(masks(:,ix),ix);
end



%Make this a format that python can read
ground_truth.bw = ground_truth.seg.bw;
ground_truth.seg = ground_truth.seg.seg;
ground_truth.Fu = ground_truth.unsuspected.Fu;
ground_truth.Su = ground_truth.unsuspected.Su;
ground_truth.unsuspectedPos = ground_truth.unsuspected.pos;
ground_truth = rmfield(ground_truth, 'unsuspected');

% check_IM = ground_truth.IM(:);
% check_IM = check_IM + ground_truth.seg*(ground_truth.activity(:,1));
% check_IM = check_IM+ground_truth.Su*(1+ground_truth.Fu(:,1));
% %check that M(:,:,1) = check_im
% if abs(sum(sum(M(:,:,1)-reshape(check_IM, size(ground_truth.IM)))))>100
%     keyboard;
% end

save([basedir filesep problemname '.mat'], 'ground_truth', 'opts', 'obs', 'Sk_masked', 'Fk', 'Su', 'Fu', 'masks');