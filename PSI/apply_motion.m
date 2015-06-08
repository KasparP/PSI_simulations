function image = apply_motion(image, shift, mode)
    %We need to be able to deal with general functions describing the warp of the sample; 
    %i.e. We shouldn't assume in our other code that all motion can be described by a matrix.
    %Using this function forces the above.
    
   
    
%TO DO: use conv2 for subpixel motion?
 %more complex motion functions?

%     xkernel = zeros(1, 3);
%     ykernel = zeros(3,1);
%     shifted =conv2(shifted, ykernel*xkernel, 'same');

switch mode
    case '2D'
        dX = shift(1);
        dY = shift(2);
        if issparse(image) %for sparse images, the (assumed square!) image is represented in the first dimension. We r
            for i = 1:size(image,2)
                tmp = reshape(image(:,i), sqrt(size(image,1)), sqrt(size(image,1)));
                tmp(max(1,1+dY):min(end, end+dY), max(1,1+dX):min(end, end+dX)) = tmp(max(1,1-dY):min(end,end-dY),  max(1,1-dX):min(end,end-dX));
                tmp([1:(max(1,1+dY)-1) end+dY+1:end],:) = 0;
                tmp(:,[1:(max(1,1+dX)-1) end+dX+1:end]) = 0;
               
                image(:,i) = reshape(tmp, size(image,1), 1);
            end
        else %for nonsparse images, treat the matrix as a 2D image X time
            image(max(1,1+dY):min(end, end+dY), max(1,1+dX):min(end, end+dX),:) = image(max(1,1-dY):min(end,end-dY),  max(1,1-dX):min(end,end-dX),:);
            image([1:(max(1,1+dY)-1) end+dY+1:end],:,:) = 0;
            image(:,[1:(max(1,1+dX)-1) end+dX+1:end],:) = 0;
        end
    case '3D'
        keyboard
        
    otherwise
        error('Invalid motion mode')
end
