function F = reconstruct_lightweight (obs,opts, SEG)
%Reconstructs F from S and P, without motion correction

%one of the segments should be the whole background.
P = opts.P';
PS = P * SEG;
D = obs.data_in;
parfor frame = 1:size(D,2)
    F(:,frame) = lsqnonneg(PS, D(:,frame));
    disp(frame)
end
end