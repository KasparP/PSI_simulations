function simulate_scope (opts_in)
%Simulates the fast 'compressed sensing' two-photon microscope

%{
TO DO
    GENERAL
        move image processing to gpuArray
        
        Incorporate prior from previous timepoint into estimate for next

        Make the projections a region in the center of IM, not all of it
            allow some seeds to move out of the projection field due to
                sample motion, and deal with not being able to recover their activity

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
opts.verbose = false;  %show some numbers and pictures during the simulation?

opts.image.fn = 'Live2-2-2013_13-19-31.tif';
opts.image.dr = [fileparts(which('simulate_scope')) filesep];
opts.framerate = 1; %frames/millisecond
opts.samplerate = 4000; %sample rate in projections (i.e. laser pulses)/millisecond; this should be the laser rep rate
opts.sim.dur = 1; %duration of simulation, milliseconds

opts.sim.amp = 4; %amplitude of signals, df/F0
opts.sim.dynamics = 'random';   %'smooth' for motion and activity varying slowly in time, or 'random' for a random bag of frames

opts.image.XYscale = 0.2; %voxel size of loaded image/standard 2P acquisition, microns
opts.image.Zscale = 1.5; %voxel size of loaded image/standard 2P acquisition, microns

opts.motion.amp.XY = 5; %amplitude of sample motion, pixels/axis
opts.motion.amp.Z = 1; %amplitude of sample motion, pixels/axis
opts.motion.speed = 20; %timescale of motion; higher numbers are slower. Only applies if opts.sim.dynamics='smooth'
opts.motion.limit = 50; %we are capping the simulated motion at this value, in pixels

opts.do3D = true; %are we simulating 2D or 3D imaging?
opts.Ptype = '4lines'; %what projection scheme are we simulating? 2lines, 4lines, etc.

opts.scope.darkrate = 5; %dark photon rate, per millisecond
opts.scope.PMTsigma = 0.5; %single photon pulse height variability of PMT/detector; sigma of gaussian, in photon eqivalents
opts.scope.readnoise = 0.3; % gaussian noise of the post-detector readout circuitry, in photon equivalents
opts.scope.brightness = 0.25; %average photons per pulse, per pixel, across the mask, at a dF/F0 of 0. This is ~0.5 photons/pulse for typical 2p imaging. We should be able to increase by a factor of ~35, because we deliver pulses ~80x slower.

%Debugging options:
opts.debug.magic_align = false; %just give the correct motion parameters to the reconstruction algorithm
opts.debug.nonoise = false; % set all noise to 0

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
tic
recon = reconstruct_imaging(obs,opts);
toc

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

while exist([opts.image.dr 'SimulationOutput' filesep filename '.mat'], 'file')
    filename = [filename '-'];
end
save([opts.image.dr 'SimulationOutput' filesep filename], 'ground_truth', 'M', 'obs', 'recon', 'opts'); 


%%
if opts.verbose
    disp('Analytics:')
    A = sim_analytics(ground_truth, M, obs, recon, opts);
end
end



