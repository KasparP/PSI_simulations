function S = segment_skel (bw, image, opts, maxseeds)
%Segment a 2D image by skeletonization

%background subtract
I2 = imtophat(medfilt2(image),strel('disk',4/opts.image.XYscale)); %radius of 4 microns

dist_thresh = opts.seg.dist_thresh/opts.image.XYscale;  %minimum distance between seeds, in pixels
nh_size = opts.seg.nh_size; %maximum radius of seed influence, in pixels. This should be >2*dist_thresh

%gaussian filter with sigma ~= dendrite radius, will help find middles of
%dendrites
h = fspecial('gaussian', 2/opts.image.XYscale, 0.8/opts.image.XYscale);
I_gauss = imfilter(I2, h);

%now skeletonize to find seed pixels
%method 1
[i1, i2] = skel_stack (bw, 0.5); %function from TREES toolbox; good for 3D but some small regions of the image are not highlighted.

%use bwselect to remove all highlighted regions
selected = false(size(bw));
selected(sub2ind(size(bw), i1,i2)) = true;
selected = imdilate(selected, ones(4*dist_thresh+1));

selected = bw & selected;
not_selected = bw & ~selected;

%now add the skeleton of the smaller regions
ns_skel = bwmorph(not_selected,'skel',Inf);
[i3, i4] = find(ns_skel);

i1 = [i1 ; i3];
i2 = [i2 ; i4];

B = I_gauss(sub2ind(size(bw),i1,i2)); %brightness of skeleton points; we'll keep the brightest ones
[~, sortorder] = sort(B, 'descend');

D = squareform(pdist([i1 i2])); %distances

%discard seed points less than dist_thresh from a brighter seed point
exclude = false(1,length(D));
for ix = 1:length(D)
    if ~exclude(sortorder(ix))
        neighbours = D(sortorder(ix),:)<dist_thresh;
        neighbours(sortorder(1:ix)) = false; %don't exclude points we've already processed
        exclude(neighbours) = true;
    end
end
i1 = i1(~exclude);
i2 = i2(~exclude);
%D = D(~exclude, ~exclude);

%show the seed points
if true
    im_disp = zeros(size(bw));
    im_disp(sub2ind(size(bw), i1,i2)) = 1;
    im_disp = conv2(im_disp, ones(3), 'same');
    figure, imshow(im_disp,[]);
end

%we are going to calculate influence of seeds over nearby pixels with a
%'flood fill' operation


bw_padded = false(size(bw) + 2*nh_size);
bw_padded(nh_size+1:end-nh_size, nh_size+1:end-nh_size) = bw;

nhood = zeros(2*nh_size+1,2*nh_size+1,length(i1));
influence = zeros(2*nh_size+1,2*nh_size+1,length(i1));
influence(nh_size+1,nh_size+1,:) = 1;
for seed = 1:length(i1)
    nhood(:,:,seed) = bw_padded((0:2*nh_size)+i1(seed), (0:2*nh_size)+i2(seed));
end

SE = [0 1 1 0; 1 1 1 1 ; 1 1 1 1 ; 0 1 1 0];
%Flood fill:
for i = 1:nh_size/2
    influence = influence + (convn(influence, SE, 'same') & nhood);
end

%convert to seg, a [#seeds x #pixels in BW] matrix
S.seg = spalloc(numel(bw), length(i1), sum(bw(:))*9); %we use a sparse matrix here
for seed = 1:length(i1)
    tmp = zeros(size(bw_padded));
    tmp((0:2*nh_size)+i1(seed),(0:2*nh_size)+i2(seed)) = influence(:,:,seed);
    tmp = tmp(nh_size+1:end-nh_size, nh_size+1:end-nh_size);
    S.seg(:,seed) = tmp(:);
end

%error checking: make sure every pixel is influenced by a seed
no_seed = ~any(S.seg(bw,:),2);
if any(no_seed)
    %some pixels have not been assigned seeds
    %make an image of these pixels
    im_noseed = false(size(bw));
    im_noseed(bw) = no_seed;
    
    bw(im_noseed) = false; %occasionally some loner pixels don't get assigned, I'll fix this later
    %figure, imshow(im_noseed,[]);
    warning('Some pixels in the mask were not assigned to a seed!')
end

%normalize
S.seg = S.seg.^2;
S.seg(bw,:) = S.seg(bw,:)./repmat(nansum(S.seg(bw,:),2), [1 length(i1)]);


%weight the segmentation by the original image intensity
S.seg(bw,:) = S.seg(bw,:).*repmat(image(bw),[1 length(i1)]);

%output the mask
S.bw = bw;
end