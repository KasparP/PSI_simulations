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
    figure, plot(residuals);
    
    %there is a strong signal in the residuals: residuals are negative
    %where unexpected signal occurs, positive along the lines
    %that pass through unexpected signal, and negative averywhere else
    
    %multiply repmats of the residuals along each axis
    pad = (size(obs.IM,1)-opts.R)/2;
    IM_res = nan([size(obs.IM) 4]);
    rot_angles = [0 270 45 135];
    for ax = 1:4 %for each projection axis
                ax_ixs = (ax-1)*opts.R + 1:ax*opts.R; %the data indices that correspond to this axis
                IM_res(:,:,ax) = nan(size(obs.IM));
                IM_res(pad+1:pad+opts.R, pad+1:pad+opts.R,ax) = repmat(residuals(ax_ixs)', opts.R,1);
                IM_res(:,:,ax) = imrotate(IM_res(:,:,ax), rot_angles(ax), 'bilinear', 'crop');
    end
    IM_res = nanmean(IM_res,3);
    figure, imshow(IM_res.^3,[])
    
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