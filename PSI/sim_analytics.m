function sim_analytics(ground_truth, M, obs, R, opts)

cmap = [ 0 0 1; gray(1024)];

%a figure of the sampled region
P_im = full(reshape(sum(opts.P,2), size(obs.IM,1), size(obs.IM,2)));
figure('Name', 'Field of view'), imshow(P_im,[])

    %backward compatibility; If you're reading this, it can probably be removed
    if ~isfield(R, 'dX')
        R.dX = round(ground_truth.motion.pos(1,:));
        R.dY = round(ground_truth.motion.pos(2,:));
    end




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
    
    keyboard
    
    figure('name','Pixel intensities'), scatter(IM_truth(Pshift_im>2.5), IM_est(Pshift_im>2.5))
    
    figure('name', 'Histogram of photon rates'), hist(obs.data_in(:,frame), 200);
    
    
    
%     zero_seeds = R.S<2*eps;
%     A = R.SEG.seg(:, zero_seeds);
%     
%     for seed = 1:size(A,2)
%         if any(A(:,seed) & P_im(R.SEG.bw)>eps)
%             imtest = zeros(size(obs.IM));
%             imtest(R.SEG.bw) = A(:,seed);
%             figure, imshow(imtest,[])
%             keyboard
%         end
%     end
end


%display Imaging parameters
if opts.verbose
    disp(['The selected image is ' 'pixels across'])
    disp('The framerate is ')
    disp('The average power is ')
    
    
    pause(1);
end

end