function [p_corr] = recon_performance(ground_truth, M, obs, R, opts)
%returns the correlation between ground truth pixel intensity and
%reconstructed pixel intensity

cmap = [ 0 0 1; gray(1024)];

P_im = full(reshape(sum(opts.P,2), size(obs.IM,1), size(obs.IM,2)));

p_corr = zeros(1,size(R,2)); %extize
for frame = 1:size(R,2)
    Pshift_im = apply_motion(P_im, [-R.dX(frame), -R.dY(frame)] );
    IM_truth = M(:,:,frame);
    %make the reconstructed image
    IM_est = obs.IM;
    IM_est(R.SEG.bw) = R.SEG.seg*R.S(:,frame);

%     IM_diff = IM_est - IM_truth; IM_diff(P_im<=eps) = 0;
%     figure('Name', 'Truth'), imshow(IM_truth,[]); colormap(cmap)
%     figure('Name', 'Reconstruction'), imshow(IM_est,[]); colormap(cmap)
%     figure('Name', 'Difference'), imshow(IM_diff,[]);

    p_corr(frame) = corr(IM_truth(Pshift_im>2.5),IM_est(Pshift_im>2.5));
end
end