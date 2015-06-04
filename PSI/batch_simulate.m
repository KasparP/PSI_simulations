function batch_simulate

%default options
opts = default_opts;

Ps = [2 4]; %projection types to simulate
Bs = [0.1 1 10 100]; %brightness levels to simulate

opts.simName = 'ProjectionTypesAndBrightness';

Pcorrs = cell(length(Ps), length(Bs));
for p_ix = 1:length(Ps)
    opts.Ptype = [int2str(Ps(p_ix)) 'lines'];
    for B_ix=1:length(Bs)
        opts.scope.brightness = Bs(B_ix);
        [ground_truth, M, obs, recon, opts] = simulate_scope(opts);
        Pcorrs{p_ix,B_ix} = recon_performance(ground_truth, M, obs, recon, opts);
    end
end
mean_corr = cellfun(@mean, Pcorrs);
err_corr = cellfun(@(x)(std(x)./sqrt(length(x))), Pcorrs);

figure, errorbar(mean_corr', err_corr')
set(gca, 'xtick',1:length(Bs), 'xticklabel', num2str(Bs'))
ylabel('Correlation-to-sample (R^2)'), xlabel('Brightness');
keyboard
