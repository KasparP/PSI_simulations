function R = reconstruct_imaging(obs,opts)
%Estimates the intensity pattern of the sample given the observations

%the expected projection:
EXP = obs.IM(:)'*opts.P; %Expected data. In final form, this will take into account the dF/F0 in previous frame, etc.

%segment the morphological image
disp('     Segmenting morphological image...')
R.SEG = segment_2D(obs.IM, opts);

%extize variables for fitting
R.F = zeros(size(R.SEG.seg,2), opts.nframes);
dX_est = zeros(1, opts.nframes);
dY_est = zeros(1, opts.nframes);
maxlag = 2*opts.motion.limit; %maximum xy motion we're willing to correct, in pixels

% 1) find the registration transform
% 2) apply registration derived motion correction to P
% pseudo-code should loook something like this....
% for frame = 1:opts.nframes,
%   P_shift(:,frame) = register(obs.IM,opts.P(:,frame), dX_est, dY_est);
% end

% now, use ADMM to estimate
% [R.F,R.S] = estimate_sources_ADMM(P_shift, obs.data_in, R.SEG.seg)


end