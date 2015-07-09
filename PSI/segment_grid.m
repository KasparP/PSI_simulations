function S = segment_grid (mask, image, maxseeds, dooptimize)

%segments an image on a grid
%the seed regions will only fall within the mask
%the number and size of seed regions to use is estimated adaptively
%to roughly ensure that #regions=#relevant projections in the '4lines'
%projection mode


ds_min = 3; %minimum downsampling factor
ds_max = 25; %maximum downsampling factor


%count the number of projections that hit the mask

nsamples = 4*size(mask,1);
if nargin>2
    nsamples = maxseeds;
end
if nargin<4
    dooptimize = true;
end

%estimate the appropriate downsampling factor
done = false;
dsfactor = ds_min;
while ~done && dsfactor<ds_max
    ds_im = downsample_bw(mask,dsfactor);
    nseeds = sum(ds_im(:));
    if nsamples>nseeds
        done = true;
    else
        dsfactor = dsfactor+1;
    end
end
disp(['Seed regions size: ' int2str(dsfactor)])

nh_size = dsfactor;

%Initialize the seed pixels ON A UNIFORM GRID:
[X,Y] = find(ds_im);
X = (X-1)*dsfactor+dsfactor/2;
Y = (Y-1)*dsfactor+dsfactor/2;

%OR: Initialize the seed pixels ON A DEFORMABLE GRID (DENSER AT HIGHER INTENSITY):

%smooth the image
%get cumulative density (capped at some maximum)
%label the pixels for the Nth crossing on each axis
%find where lines cross

%optimize location of seeds
[Xbw, Ybw] = find(mask);
if dooptimize
    [X,Y] = optimize_seeds(X,Y,Xbw,Ybw);
    
    %0. remove points that are very crowded
    cutoff = dsfactor/2; %we'll prune points that are closer together than this number
    [X,Y] = prune_small(X,Y, cutoff);
    
    %recalculate assignments
    minind = knnsearch([X Y], [Xbw Ybw]);
    
    %do some things to improve the segmentations:
    %1. break points that represent multiple separate regions into points
    %for each region
    [X,Y] = split_regions(X,Y,Xbw,Ybw, minind);
    
    %recalculate assignments
    minind = knnsearch([X Y], [Xbw Ybw]);
    
    %2. split points that are covering a large number of pixels into
    %multiple points
    [X,Y] = split_big(X,Y,Xbw,Ybw, minind);
    
    %reoptimize the seeds
    [X,Y] = optimize_seeds(X,Y,Xbw,Ybw);
end
%recalculate assignments
minind = knnsearch([X Y], [Xbw Ybw]);

%     figure, imshow(mask);
%     hold on, scatter(Y,X);
%     keyboard


% %get the local neighbourhoods for each seed point
% nhood = zeros(dsfactor,dsfactor,length(X));
% for seed = 1:length(X)
%     nhood(:,:,seed) = bw_padded(nh_size+ceil(0.1+(0:dsfactor-1)+X(seed)-dsfactor/2), nh_size+ceil(0.1+(0:dsfactor-1)+Y(seed)-dsfactor/2));
%     s  = regionprops(nhood(:,:,seed),'centroid');
%     X(seed) = s.Centroid(1) + X(seed) - dsfactor/2;
%     Y(seed) = s.Centroid(2) + Y(seed) - dsfactor/2;
% end

%assemble the complete map
%convert to seg, a [#seeds x #pixels in BW] matrix

% S.seg = spalloc(numel(mask),length(X), sum(mask(:)));
% for i = 1:length(Xbw)
%    S.seg(sub2ind(size(mask),Xbw(i), Ybw(i)), minind(i)) = 1;
% end

indsX = sub2ind(size(mask),Xbw, Ybw);
indsY = minind;
vals = ones(1,sum(mask(:)));
S.seg = sparse(indsX,indsY,vals,numel(mask),length(X));
S.bw = mask;

% %accumulate the indexes for the sparse matrix
% indsX = zeros(1, sum(mask(:)));
% indsY = zeros(1, sum(mask(:)));
% vals = ones(1,sum(mask(:)));
% ix = 0;
% for seed = 1:length(X)
%     X_ixs = Xbw(minind==seed);
%     Y_ixs = Ybw(minind==seed);
%     indsX(ix+1:ix+length(X_ixs)) = sub2ind(size(mask),X_ixs, Y_ixs);
%     indsY(ix+1:ix+length(X_ixs)) = seed;
%     ix = ix+length(X_ixs);
% end
%build the sparse matrix



end

function [X,Y] = optimize_seeds(X,Y,Xbw,Ybw)
XY_im = zeros(ceil(max(max(X),max(Xbw))),ceil(max(max(Y),max(Ybw))));
XY_im(sub2ind(size(XY_im),round(X),round(Y))) = 1:length(X);

Xold = inf(size(X)); Yold = inf(size(X));
iters = 0;
minind = zeros(1,length(Xbw));
while iters<20 && (any(abs(Xold-X)>0.1) || any(abs(Yold-Y)>0.1))
    tic
    iters = iters+1
    Xold = X; Yold = Y;
    
    minind = knnsearch([X Y], [Xbw Ybw]);
    
    %reset the seed points to the center of mass of their associated pixels
    for i = 1:length(X)
        if isnan(mean(Xbw(minind==i)))
            keyboard
        end
        X(i) = mean(Xbw(minind==i));
        Y(i) = mean(Ybw(minind==i));
    end
    toc
end
end

function [X,Y] = prune_small(X,Y, cutoff)
    %calculate all distances between seed points
    D = squareform(pdist([X,Y]));
    N = sum(D<cutoff);
    
    [sorted, sortorder] = sort(N);
    
    first = find(sorted>1,1,'first');
    
    if ~isempty(first)
        %discard seed points less than dist_thresh from a brighter seed point
        exclude = false(1,length(D));
        for ix = first:length(D)
            if ~exclude(sortorder(ix))
                neighbours = D(sortorder(ix),:)<cutoff;
                neighbours(sortorder(1:ix)) = false; %don't exclude points we've already processed
                exclude(neighbours) = true;
            end
        end
        X = X(~exclude);
        Y = Y(~exclude);
        %sort seed points according to the number of distances they have less than cutoff
    end

end

function [X,Y] = split_regions(X,Y,Xbw,Ybw, minind)
    for i = 1:length(X)
        Xs = Xbw(minind==i); Ys = Ybw(minind==i);
        minX = min(Xs); maxX = max(Xs);
        minY = min(Ys); maxY = max(Ys);
        
        window = false(maxX-minX+1, maxY-minY+1);
        window(sub2ind(size(window), Xs-minX+1, Ys-minY+1)) = true;
        L = bwlabel(window, 4);
        if max(L(:))>1
            X(i) = sum((sum(L==1,2).*(1:size(window,1))')/sum(L(:)==1)) + minX -1;
            Y(i) = sum((sum(L==1,1).*(1:size(window,2)))/sum(L(:)==1)) + minY -1;
            for j = 2:max(L(:))
                X(end+1) = sum((sum(L==j,2).*(1:size(window,1))')/sum(L(:)==j)) + minX -1;
                Y(end+1) = sum((sum(L==j,1).*(1:size(window,2)))/sum(L(:)==j)) + minY -1;
            end
        end
    end
end

function [X,Y] = split_big(X,Y,Xbw,Ybw, minind)
    [numpoints]= histc(minind, 1:length(X));
    M = median(numpoints);
    
    toobig = find(numpoints'>(2*M));
    for i = toobig
        [vecs, vals] = eig(cov(Xbw(minind==i),Ybw(minind==i)));
        [maxval, maxind] = max(diag(vals));

        X(end+1) = X(i)+0.1*vecs(1,maxind);
        Y(end+1) = Y(i)+0.1*vecs(2,maxind);
        X(i) = X(i)-0.1*vecs(1,maxind);
        Y(i) = Y(i)-0.1*vecs(2,maxind);
    end
end

function IM2 = downsample_bw(IM, n)
    IM2 = false([ceil(size(IM)/n)]);
    for i = 1:n
        IM2tmp = IM(i:n:end,i:n:end);
        IM2(1:size(IM2tmp,1),1:size(IM2tmp,2)) = IM2(1:size(IM2tmp,1),1:size(IM2tmp,2)) | IM2tmp;
    end
end
