function image = apply_motion(image, shift)
    %We need to be able to deal with general functions describing the warp of the sample; 
    %i.e. We shouldn't assume in our other code that all motion can be described by a matrix.
    %Using this function forces the above.
    
    %more complex motion functions?
    
    
%TO DO: use conv2 for subpixel motion?
%     xkernel = zeros(1, 3);
%     ykernel = zeros(3,1);
%     shifted =conv2(shifted, ykernel*xkernel, 'same');

%image can be a BIG matrix. Avoid memory issues by shifting it in place?
    dX = shift(1);
    dY = shift(2);
    image(max(1,1+dY):min(end, end+dY), max(1,1+dX):min(end, end+dX),:) = image(max(1,1-dY):min(end,end-dY),  max(1,1-dX):min(end,end-dX),:);
    image([1:(max(1,1+dY)-1) end+dY+1:end],:,:) = 0;
    image(:,[1:(max(1,1+dX)-1) end+dX+1:end],:) = 0;
    
end
