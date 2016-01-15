function GT = simulate_sample(opts)
%Simulates the 'Ground Truth' parameters GT
%   GT.motion: the sample motion
%   GT.seg: The segmentation of the image into seed regions
%   GT.activity: The activity trace of each seed

%read in an image which represents the neuron
disp('    Reading image data...')
if opts.do3D
    A = tiffread2([opts.image.dr opts.image.fn]);
    GT.IM = cell2mat(reshape({A.data},1,1,[]));
    
    %normalize the image to have the correct brightness
    %find regions inside the neuron
    %background subtract
    
    %threshold = 2 standard deviations of each plane?
    bw = false(size(GT.IM));
    I2 = zeros(size(GT.IM));
    for plane = 1:size(GT.IM,3)
        I2tmp = imtophat(medfilt2(GT.IM(:,:,plane)),strel('disk',4/opts.image.XYscale));
        thresh = prctile(I2tmp(:), 99)*0.5; %need a better thresholding method
        bw(:,:,plane) = I2tmp>thresh;
        I2(:,:,plane) = I2tmp;
    end
    mean_int = mean(double(I2(bw)));
    GT.IM = double(GT.IM).*opts.scope.brightness/mean_int;
    
    clear I2tmp I2 bw%free up memory
else
    imnums = 41:45;
    A = tiffread2([opts.image.dr opts.image.fn], min(imnums), max(imnums));
    GT.IM = max(cell2mat(reshape({A.data},1,1,[])),[],3);
    
    %normalize the image to have the correct brightness
    %find regions inside the neuron
    %background subtract
    I2 = imtophat(medfilt2(GT.IM),strel('disk',4/opts.image.XYscale)); %radius of 4 microns
    thresh = prctile(I2(:), 99)*0.5;
    bw = I2>thresh;
    mean_int = mean(double(I2(bw)));
    GT.IM = double(GT.IM)*opts.scope.brightness/mean_int;
end

%MOTION
GT.motion.pos = nan(3,opts.sim.dur);
for dim = 1:3
    if dim<3
        motion = opts.motion.amp.XY;
    else
        motion = opts.motion.amp.Z;
    end
    if strcmpi(opts.sim.dynamics, 'smooth')
        series = smooth(max(-opts.motion.limit/motion, min(opts.motion.limit/motion, sqrt(opts.motion.speed).*randn(1, 2*opts.motion.speed+opts.sim.dur))), opts.motion.speed);
        GT.motion.pos(dim,:) = motion .* series(opts.motion.speed+1:(end-opts.motion.speed));
    elseif strcmpi(opts.sim.dynamics, 'random')
        GT.motion.pos(dim,:) = max(-opts.motion.limit, min(opts.motion.limit, motion .* randn(1, opts.sim.dur)));
    else
        error('Option [opts.sim.dynamics] should be set to either ''smooth'' or ''random''');
    end
end

%SEGMENTATION
%A segmentation is a sparse npixels x nsegments array. Nonzero elements
%are considered 'inside' the cell
if opts.do3D
    GT.seg = segment_3D(GT.IM, opts);
else
    GT.seg = segment_2D(GT.IM, opts);
end

GT.nseeds = size(GT.seg.seg, 2);

%ACTIVITY
%generate a random smooth non-negative timeseries for each 'ground
%truth' seed
kernel = normpdf(-50*opts.framerate:50*opts.framerate, 0, 10*opts.framerate); %kernel in frames
kernel = kernel./max(kernel);
if strcmpi(opts.sim.dynamics, 'smooth')
    GT.activity = opts.sim.amp .* convn(poissrnd(1/(50*opts.framerate), GT.nseeds, opts.nframes), kernel ,'same');
else %random
    GT.activity = opts.sim.amp .* convn(poissrnd(1/(50*opts.framerate), GT.nseeds, max(500*length(kernel), opts.nframes)), kernel ,'same');
    GT.activity = GT.activity(:, randperm(size(GT.activity,2), opts.nframes));
end


%UNSUSPECTED ACTIVITY
if opts.sim.unsuspected.N
    GT.unsuspected.pos = nan(2,opts.sim.unsuspected.N); %extize
   
    %find a region of the image to put the 'unsuspected' activity, outside the morphological mask 
    valid_pos = ~imdilate(GT.seg.bw, strel('disk', 0.8/opts.image.XYscale));
    [X,Y] = meshgrid(1:size(valid_pos,1));
    radius = size(valid_pos,1)/4;
    mask = sqrt((X-size(valid_pos,1)/2).^2 + (Y-size(valid_pos,1)/2).^2)<radius;
    valid_pos = valid_pos & mask;
    
    %pick a random spot within the valid region
    N=1;
    while N<=opts.sim.unsuspected.N
         xy= randi(size(valid_pos,1), 1,2);
         GT.unsuspected.pos(:,N) = xy;
        if valid_pos(xy(1),xy(2))
            N = N+1;
            valid_pos(xy(1),xy(2)) = false;
        end
    end
    
    %generate Su
    un_kernel = double(getnhood(strel('disk',ceil(1.6/opts.image.XYscale))));
    un_radius = floor(size(un_kernel,1)/2);
    GT.unsuspected.Su = zeros(numel(GT.IM), opts.sim.unsuspected.N);
    for n = 1:opts.sim.unsuspected.N
        S = zeros(size(GT.IM));
        xs = (-un_radius:un_radius) + GT.unsuspected.pos(1,n);
        ys = (-un_radius:un_radius) + GT.unsuspected.pos(2,n);
        S(xs,ys) =  un_kernel*opts.scope.brightness*opts.sim.unsuspected.amp;
        GT.unsuspected.Su(:,n) = S(:);
    end
    
    if strcmpi(opts.sim.dynamics, 'smooth')
        GT.unsuspected.Fu = opts.sim.amp .* convn(poissrnd(1/(50*opts.framerate), opts.sim.unsuspected.N, opts.nframes), kernel ,'same');
    else %random
        GT.unsuspected.Fu = opts.sim.amp .* convn(poissrnd(1/(50*opts.framerate), opts.sim.unsuspected.N, max(500*length(kernel), opts.nframes)), kernel ,'same');
        GT.unsuspected.Fu =  GT.unsuspected.Fu(:, randperm(size(GT.unsuspected.Fu,2),opts.nframes));
    end
end

end