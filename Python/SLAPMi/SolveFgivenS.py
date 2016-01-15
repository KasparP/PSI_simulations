import lsqlin
import numpy as np


def SolveFgivenS (Y, P0, S, motion = None)

    for frame in range(0,Y.shape[1])
    PS = P(frame)

    ret = lsqlin.lsqnonneg(PS, Y, {'show_progress': False})
   #S is the concatenation [Sk Su]







def P (frame)
    return P0








F = reconstruct_lightweight (obs,opts, SEG)
%Reconstructs F from S and P, without motion correction

%one of the segments should be the whole background.
P = opts.P';
PS = P * SEG;
D = obs.data_in;
parfor frame = 1:size(D,2)
    F(:,frame) = lsqnonneg(PS, D(:,frame));
    disp(frame)
end



F = reconstruct_lightweight (obs,opts, SEG)
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