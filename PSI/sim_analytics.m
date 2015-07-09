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
    Pshift_im = apply_motion(P_im, [-R.dX(frame), -R.dY(frame)], '2D');
    IM_truth = M(:,:,frame);
    %make the reconstructed image
    IM_est = obs.IM;
    IM_est(R.SEG.bw) = R.SEG.seg(R.SEG.bw,:)*R.S(:,frame);

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
    IM_est(R.SEG.bw) = R.SEG.seg(R.SEG.bw,:)*R.S(:,frame);
    IM_shift = apply_motion(IM_est, [R.dX(frame), R.dY(frame)], '2D');
    
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
    
    IM_seg = apply_motion(R.SEG.seg, [R.dX(frame), R.dY(frame)], '2D'); %IM_seg, shifted
    P_shift = opts.P' * IM_seg;
    
    %estimate with nonnegative constraint
    S_neg = lsqnonneg(P_shift(select_pixels,:), D_neg);
    
    %reconstruct image of negative residuals
    IM_neg = zeros(size(obs.IM));
    IM_neg(R.SEG.bw) = R.SEG.seg(R.SEG.bw,:)*S_neg;
    
    figure('Name','Reconstructed Negative Residuals (estimate of IM_diff from observed data only)'), imshow(IM_neg,[]);
    
    %Now, reproject the negative residuals and add them to the data
    res_corrected = residuals + P_shift*S_neg; %corrected residuals, will be used to estimate locations of unknown sources
    
    %create a mask of the entire background space
    mask_bg = ~R.SEG.bw;
    maxseeds = sum((opts.P'*mask_bg(:)) >0.5);
    S_bg = segment_grid(mask_bg,[],inf, false);
    
    IM_seg_bg = apply_motion(S_bg.seg, [R.dX(frame), R.dY(frame)], '2D'); %IM_seg, shifted
    P_shift_bg = opts.P' * IM_seg_bg;
    
    %remove the seeds that don't overlap the imaging region
    select = any(P_shift_bg,1);
    
    [lambda_max] = find_lambdamax_l1_ls_nonneg(P_shift_bg(:,select)',res_corrected);
    %solve as an L1-regularized least squares problem
    [x,status,history] = l1_ls_nonneg(P_shift_bg(:,select),res_corrected,lambda_max*1E-2);
    
    x2 = zeros(size(S_bg.seg,2),1);
    x2(select) = x;
    
    %reconstruct the image
    reconstructed = reshape(IM_seg_bg*x2, size(obs.IM,1), size(obs.IM,2));
    
    %smooth
    h = fspecial('gaussian', 4, 4);
    recon2 = imfilter(reconstructed, h);
    addmask = recon2>5*std(recon2(:));
    addmask = imopen(addmask, strel('disk',2));
    
    figure('Name', 'Estimate of locations of unexpected activity'), imshow(addmask)
    
    %we will be adding points to the segmentation
    numseeds = 3* sum(addmask(:))/sum(R.SEG.bw(:)) * size(R.SEG.seg,2);
    add_seg = segment_grid(addmask,[], numseeds, true);
    
    %to do: redo reconstruction with expanded segmentation
end
end