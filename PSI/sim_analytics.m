function sim_analytics(ground_truth, M, obs, R, opts)

%ANALYSES
do_basic = true;
do_residuals = true;

cmap = [ 0 0 1; gray(1024)];

%a figure of the sampled region
P_im = full(reshape(sum(opts.P,2), size(obs.IM,1), size(obs.IM,2)));
figure('Name', 'Field of view'), imshow(P_im,[])


%BASIC ANALYSIS
if do_basic
for frame = 1:size(R,2)
    Pshift_im = apply_motion(P_im, [-R.dX(frame), -R.dY(frame)] );
    IM_truth = M(:,:,frame);
    %make the reconstructed image
    IM_est = obs.IM;
    IM_est(R.SEG.bw) = R.SEG.seg*R.S(:,frame);

    figure('Name', 'Truth'), imshow(IM_truth,[]); colormap(cmap)
    figure('Name', 'Reconstruction'), imshow(IM_est,[]); colormap(cmap)
    
    IM_diff = IM_est - IM_truth; IM_diff(P_im<=eps) = 0;
    figure('Name', 'Difference'), imshow(IM_diff,[]);
    figure('name', 'Pixel intensities'), scatter(IM_truth(Pshift_im>2.5), IM_est(Pshift_im>2.5))
    figure('name', 'Histogram of photon rates'), hist(obs.data_in(:,frame), 200);
end
end


    
%RESIDUALS - might be used to learn the segmentation from the imaging data
if do_residuals
for frame = 1:size(R,2)    
    IM_est = obs.IM;
    IM_est(R.SEG.bw) = R.SEG.seg*R.S(:,frame);
    IM_shift = apply_motion(IM_est, [R.dX(frame), R.dY(frame)]);
    
    %estimate residuals in the projection space
    data_est = IM_shift(:)'*opts.P;
    
    residuals = obs.data_in(:,frame) - data_est';
    %adjust for constant offset in the residuals?
    
    figure('Name', 'Residuals of Reconstruction'), plot(residuals);
    
    %there is a strong signal in the residuals: residuals are negative
    %where unexpected signal occurs, positive along the lines
    %that pass through unexpected signal, and negative averywhere else
    
    
    %Take negative residuals and assign them to the known seeds
    %The goal is to 'undo' some of the mixing of the unexpected
    %signal, which increases signals in first-order related seeds, and
    %decreases signal in the second-order seeds.
    %this mixes the decreased signal back onto the first order seeds, which
    %will then be used to estimate the location of the source.
    
    select_pixels = residuals<0; %have some cutoff, -X, set by the noise?
    D_neg = -residuals(select_pixels);
    
    IM_seg = zeros(size(obs.IM,1),size(obs.IM,2), size(R.SEG.seg,2));%extize
    IM_seg(repmat(R.SEG.bw,1,1,size(R.SEG.seg,2))) = R.SEG.seg; %IM_seg is a BIG matrix. no better general way? make sparse?
    IM_seg = apply_motion(IM_seg, [R.dX(frame), R.dY(frame)]); %IM_seg, shifted
    P_shift = opts.P' * reshape(IM_seg, [size(IM_seg,1)*size(IM_seg,2), size(R.SEG.seg,2)]);
    
    %estimate with nonnegative constraint
    S_neg = lsqnonneg(P_shift(select_pixels,:), D_neg);
    
    %reconstruct image of negative residuals
    IM_neg = zeros(size(obs.IM));
    IM_neg(R.SEG.bw) = R.SEG.seg*S_neg;
    
    figure('Name','Reconstructed Negative Residuals'), imshow(IM_neg,[]);
    keyboard
    
    %Now, reproject the negative residuals and add them to the data
    res_corrected = residuals + P_shift*S_neg; %corrected residuals, will be used to estimate locations of unknown sources
    
    %create a mask of the entire background space
    mask_bg = ~R.SEG.bw;
    S_bg = segment_grid(mask_bg);
    IM_seg_bg = zeros(size(obs.IM,1),size(obs.IM,2), size(S_bg.seg,2));%extize
    IM_seg_bg(repmat(S_bg.bw,1,1,size(S_bg.seg,2))) = S_bg.seg; %IM_seg is a BIG matrix. no better general way? make sparse?
    IM_seg_bg = apply_motion(IM_seg_bg, [R.dX(frame), R.dY(frame)]); %IM_seg, shifted
    P_shift_bg = opts.P' * reshape(IM_seg_bg, [size(IM_seg_bg,1)*size(IM_seg_bg,2), size(S_bg.seg,2)]);
    
    %solve as an L1-regularized least squares problem
    keyboard
    
    
    
    %multiply repmats of the residuals along each axis
    pad = (size(obs.IM,1)-opts.R)/2;
    IM_res = nan([size(obs.IM) 4]);
    rot_angles = [0 270 45 135];
    for ax = 1:4 %for each projection axis
                ax_ixs = (ax-1)*opts.R + 1:ax*opts.R; %the data indices that correspond to this axis
                IM_res(:,:,ax) = nan(size(obs.IM));
                IM_res(pad+1:pad+opts.R, pad+1:pad+opts.R,ax) = repmat(res_corrected(ax_ixs)', opts.R,1);
                IM_res(:,:,ax) = imrotate(IM_res(:,:,ax), rot_angles(ax), 'bilinear', 'crop');
    end
    IM_res = nanmean(IM_res,3);
    figure('name','Triangulation of unsuspected sources'), imshow(IM_res.^3,[])
    
    %how to threshold image for candidate sites?
        %simple threshold on cube of average image
        %demand that all projections > median?
        %demand that 3/4 projections >median?
    
    %fit the residuals with matrix division optimized for sparsity
    %Generate a new set of bases
    
    %fit the positive and negative residuals separately, with nonnegative
    %fitting
    %positive residuals should lie outside the morphological mask
    
    keyboard 
end
end