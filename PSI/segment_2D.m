function S = segment_2D (I, opts)
    %For now, a quick and dirty method: local thresholding, then throw away
    %small spatial outliers
    
    %Thresholding
    %background subtraction:
    I2 = imtophat(medfilt2(I),strel('disk',4/opts.image.XYscale)); %radius of 4 microns
    
    %global threshold on the background-subtracted image
    thresh = prctile(I2(:), 99)*0.5;
    %generate mask
    bw = I2>thresh; 
    bw = imdilate(bw, strel('diamond',1));
    %remove regions of size less than 40 pixels
    bw = bwareaopen(bw, 40);
    if true
        figure, imshow(bw)
    end
    
    S = segment_skel(bw, I, opts);
end