function [P,R] = generate_projections(imagesize, opts)
%generates a projection matrix (i.e. the distribution of excitation at the time of each laser
%pulse/intensity measurement) used either in simulation of the imaging system or for reconstruction of data
%different projections can be selected with the 'method' argument

%Simulations indicate the using 4 lines is equivalent to collecting ~10x as many
%photons while scanning 2 lines. 4 lines is ~0.5 times as good as collecting each pixel
%separately, per photon.

switch opts.Ptype
    case '4lines'
        %line segments crossing the sample along 4 axes spaced by 45 degreed
        if imagesize(1)~=imagesize(2) || mod(imagesize(1),2)
            error('The 4lines algorithm currently assumes a square image plane with a multiple of 4 pixels');
        end
        
        R = imagesize(1)/2; %resolution; i.e., the number of pixels across the projection
        P = spalloc(imagesize(1)*imagesize(2), 4*R, 2*imagesize(1)*imagesize(2));
       
        
        pad = (imagesize(1)-R)/2;
        
        for a = 1:R
            ix_1 = pad+(1:R);
            ix_2 = (pad+a)*ones(1,R);
            v = ones(1,R);
            
            template = sparse(ix_1,ix_2,v, imagesize(1),imagesize(2));
            R45 = imrotate(template,45, 'bilinear', 'crop');
            R45 = sum(template(:))*R45./sum(R45(:));
            
            P(:,a) = reshape(template,[],1);
            P(:,R+a) = reshape(sparse(ix_2,ix_1,v, imagesize(1),imagesize(1)),[],1);
            P(:,2*R+a) = reshape(sparse(R45),[],1);
            P(:,3*R+a) = reshape(sparse(rot90(R45)),[],1);
            
        end
    case '2lines'
        %line segments crossing the sample along the 2 cardinal axes,
        %NOTE: Currently each axis is scanned twice to better compare with 4lines!
        if imagesize(1)~=imagesize(2) || mod(imagesize(1),2)
            error('The 4lines algorithm currently assumes a square image plane with a multiple of 2 pixels');
        end
        
        R = imagesize(1)/2; %resolution; i.e., the number of pixels across the projection
        P = spalloc(imagesize(1)*imagesize(2), 4*R, 2*imagesize(1)*imagesize(2));
       
        
        pad = (imagesize(1)-R)/2;
        
        for a = 1:R
            ix_1 = pad+(1:R);
            ix_2 = (pad+a)*ones(1,R);
            v = ones(1,R);
            
            template = sparse(ix_1,ix_2,v, imagesize(1),imagesize(2));
            P(:,a) = reshape(template,[],1);
            P(:,R+a) = reshape(sparse(ix_2,ix_1,v, imagesize(1),imagesize(1)),[],1);
            P(:,2*R+a) = P(:,a);
            P(:,3*R+a) = reshape(sparse(ix_2,ix_1,v, imagesize(1),imagesize(1)),[],1);
        end
        
        
    otherwise
        error('Invalid projection method selected')
end
end