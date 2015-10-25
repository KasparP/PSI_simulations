function [ground_truth, M, obs, recon, opts] = simulate_scope (opts_in)
%Simulates the fast 'compressed sensing' two-photon microscope

%{
TO DO
    GENERAL
        move image processing to gpuArray
        
        Incorporate prior from previous timepoint into estimate for next

    SEGMENTATION
        Alternative skeletonization?
            Want to ensure seed pixels at ends of processes, and a minimum
                density
            easily done in 2D /w bwmorph: skeletonize, select branchpoints
                and endpoints as seeds, remove all pixels within radius X  
                of these from skeleton, repeat selection. 3D?
        Base segmentation on neuron tracing? Hard to do well.
        Alter 'flood-fill' method to depend on intensity of intervening
            pixels?
        'Validate' segmentation by generating data one way and recovering
            another
    
    SIMULATION
        Generate correlated activity

    RECONSTRUCTION
        %Is least squares best cost function? KL divergence?
%}

%default options
opts = default_opts;

if nargin %update any options passed as argument
    flds = fieldnames(opts_in);
    for fld_num = 1:length(flds)
        opts.(flds{fld_num}) = opts_in.(flds{fld_num});
    end
end

%some aliases that make things simpler
opts.nframes = opts.sim.dur*opts.framerate;

%warnings if the simulation is using a debug mode
if opts.debug.magic_align 
	disp('WARNING: Using DEBUG: MAGIC ALIGN mode')
end
if opts.debug.nonoise
    disp('WARNING: Using DEBUG: NO NOISE mode')
    opts.scope.darkrate = 0; %dark photon rate, per millisecond
    opts.scope.PMTsigma = 0; %single photon pulse height variability of PMT/detector; sigma of gaussian, in photon eqivalents
    opts.scope.readnoise = 0;
end


%%
%Begin Simulation
disp('Generating ground truth.')
tic
ground_truth = simulate_sample(opts);
toc

%%
disp('Generating projections.')
tic
[opts.P, opts.R]= generate_projections(size(ground_truth.IM), opts);
toc
%%

disp('Simulating imaging.')
tic
[obs, M] = simulate_imaging(ground_truth, opts);
toc
%%

%if debugging, we'll use some of the ground truth to skip estimation steps in the reconstruciton
if opts.debug.magic_align
    opts.debug.GT.motion = ground_truth.motion;
end

disp('Performing reconstruction.')
recon = reconstruct_imaging_ADMM(obs,opts,ground_truth);
% tic
% recon = reconstruct_imaging(obs,opts);
% toc

%%
%save the output
nametags = '';
if opts.debug.magic_align || opts.debug.nonoise
    nametags = [nametags 'DEBUG-'];
end
filename = ['sim-' nametags datestr(clock,'mm-dd_HH-MM')];

if ~exist([opts.image.dr 'SimulationOutput'], 'dir')
    mkdir([opts.image.dr 'SimulationOutput'])
end
if ~exist([opts.image.dr 'SimulationOutput' filesep opts.simName], 'dir')
    mkdir([opts.image.dr 'SimulationOutput' filesep opts.simName])
end

while exist([opts.image.dr 'SimulationOutput' filesep opts.simName filesep filename '.mat'], 'file')
    filename = [filename '-'];
end
save([opts.image.dr 'SimulationOutput' filesep opts.simName filesep filename], 'ground_truth', 'M', 'obs', 'recon', 'opts'); 


%%
if opts.verbose
    disp('Analytics:')
    sim_analytics(ground_truth, M, obs, recon, opts);
end
end



