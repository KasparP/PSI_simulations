function simulate_precomp
startup
obs = []; opts = []; ground_truth = [];
load('PRECOMP_nonoise')

R = reconstruct_imaging_ADMM(obs,opts,10000, 40, ground_truth);

mask = ground_truth.seg.bw;

T = nan(size(mask));

for i = 1:size(R.Sk,2)
    T(mask) = R.Sk(:,i);
    figure, imshow(T,[]);
    input('hit enter')
end


keyboard