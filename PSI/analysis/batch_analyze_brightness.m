function batch_analyze_brightness

[fns, dr] = uigetfile('*.mat', 'Select your savefiles', 'Multiselect', 'on');
if ~iscell(fns) %uigetfile doesn't return a cell array for a single filename
    fns = {fns};
end

ground_truth = []; M= []; obs = []; recon = []; opts = []; %these variables will be loaded below


%each file is simulated at a different brightness, but other opts are the
%same
for fnum = 1:length(fns)
    load([dr fns{fnum}]);
    
    B = opts.scope.brightness;
    P_im = full(reshape(sum(opts.P,2), size(obs.IM,1), size(obs.IM,2)));
    
    %backward compatibility; If you're reading this, it can probably be removed
    if ~isfield(recon, 'dX')
        recon.dX = round(ground_truth.motion.pos(1,:));
        recon.dY = round(ground_truth.motion.pos(2,:));
    end
    
    
%We will accrue pixel intensities
    pixels_exp = cell(1,size(recon.S,2));
    pixels_obs = cell(1,size(recon.S,2));
    
    for frame = 1:size(recon.S,2)
        
        Pshift_im = apply_motion(P_im, [-recon.dX, -recon.dY] );
        IM_truth = M(:,:,frame);
        %make the reconstructed image
        IM_est = obs.IM;
        IM_est(recon.SEG.bw) = recon.SEG.seg*recon.S(:,frame);
        
        %figure('Name', 'Truth'), imshow(IM_truth,[]);
        %figure('Name', 'Reconstruction'), imshow(IM_est,[]);
        
        IM_diff = IM_est - IM_truth; IM_diff(P_im<=eps) = 0;
        %figure('Name', 'Difference'), imshow(IM_diff,[]);
        
        
        figure('name','Pixel intensities'), scatter(IM_truth(Pshift_im>2.5), IM_est(Pshift_im>2.5))
        
        %figure('name', 'Histogram of photon rates'), hist(obs.data_in(:,frame), 200);
        
        %accumulate the pixel intensities
        pixels_exp{frame} = IM_truth(Pshift_im>2.5);
        pixels_obs{frame} = IM_est(Pshift_im>2.5);
        
        keyboard
    end
    
    pixels_exp = cell2mat(pixels_exp');
    pixels_obs = cell2mat(pixels_obs');
    errs = pixels_obs-pixels_exp;
    
    

    
    binsize = 20;
    nbins = ceil(max(pixels_exp)/binsize);
    edges = 0:binsize:nbins*binsize;
    stdev = zeros(1,nbins);
    for bin = 1:nbins
        select = pixels_exp>=edges(bin) & pixels_exp<edges(bin+1);
        stdev(bin) = std(errs(select));
        
    end
    
    %expected standard deviation
    photons_collected = [0:edges(end)];
    stdev_exp = sqrt(photons_collected + opts.scope.PMTsigma.^2 .* photons_collected  +  opts.scope.readnoise.^2);
    
    figure, plot([0:edges(end)], stdev_exp, 'linewidth', 2);
    hold on, scatter(edges(1:end-1)+binsize/2, stdev, 'markeredge', 'r', 'markerface', 'none', 'linewidth', 2)
    
    
    keyboard
end















