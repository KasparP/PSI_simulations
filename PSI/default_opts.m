function opts = default_opts (opts_in)

%default options for PSI scope simulations
opts.verbose = true;  %show some numbers and pictures during the simulation?

opts.image.fn = 'Live2-2-2013_13-19-31.tif';
opts.image.dr = [fileparts(which('simulate_scope')) filesep];
opts.framerate = 1; %frames/millisecond
opts.samplerate = 4000; %sample rate in projections (i.e. laser pulses)/millisecond; this should be the laser rep rate

opts.sim.dur = 1000; %duration of simulation, milliseconds
opts.sim.amp = 8; %amplitude of signals, df/F0
opts.sim.dynamics = 'smooth';   %'smooth' for motion and activity varying slowly in time, or 'random' for a random bag of frames with the same statistics as 'smooth'
opts.sim.unsuspected.N = 3; % an integer >=0, adds this number of 'unsuspected' signals to the image in regions outside the morphological mask, to simulate failures in segmentation due to weak morphological signal
opts.sim.unsuspected.amp = 1;

opts.nframes = opts.sim.dur*opts.framerate;

opts.image.XYscale = 0.2; %voxel size of loaded image/standard 2P acquisition, microns
opts.image.Zscale = 1.5; %voxel size of loaded image/standard 2P acquisition, microns

opts.motion.amp.XY = 0; %amplitude of sample motion, pixels/axis
opts.motion.amp.Z = 0; %amplitude of sample motion, pixels/axis
opts.motion.speed = 50; %timescale of motion; higher numbers are slower. Only applies if opts.sim.dynamics='smooth'
opts.motion.limit = 10; %we are capping the simulated motion at this value, in pixels

opts.seg.nh_size = 30; %the region around each seed to look for related pixels
opts.seg.dist_thresh = 0.8; %used to select the number of seeds K to initialize

opts.do3D = false; %are we simulating 2D or 3D imaging?
opts.Ptype = '4lines'; %what projection scheme are we simulating? 2lines, 4lines, etc.

opts.scope.brightness = 2; %average photons per pulse, per pixel, across the mask, at a dF/F0 of 0. This is ~0.5 photons/pulse for typical 2p imaging. We should be able to increase by a factor of ~35, because we deliver pulses ~80x slower.

opts.scope.detectortype = 'HPD';  %HPD , PMT , or MPPC
if nargin && isfield(opts_in, 'scope') && isfield(opts_in.scope, 'detectortype')
    opts.scope.detectortype = opts_in.scope.detectortype;
end
    
switch opts.scope.detectortype
    case 'HPD'
        opts.scope.darkrate = 5; %dark photon rate, per millisecond
        opts.scope.PMTsigma = 0.1; %single photon pulse height variability of PMT/detector; sigma of gaussian, in photon eqivalents
        opts.scope.readnoise = 1; % gaussian noise of the post-detector readout circuitry, in photon equivalents
    case 'PMT'
        opts.scope.darkrate = 5; %dark photon rate, per millisecond
        opts.scope.PMTsigma = 1; %single photon pulse height variability of PMT/detector; sigma of gaussian, in photon eqivalents
        opts.scope.readnoise = 0.3; % gaussian noise of the post-detector readout circuitry, in photon equivalents
    case 'MPPC'
        opts.scope.darkrate = 100; %dark photon rate, per millisecond
        opts.scope.PMTsigma = 0.1; %single photon pulse height variability of PMT/detector; sigma of gaussian, in photon eqivalents
        opts.scope.readnoise = 0.3; % gaussian noise of the post-detector readout circuitry, in photon equivalents
end

%Debugging options:
opts.debug.magic_align = true; %just give the correct motion parameters to the reconstruction algorithm
opts.debug.nonoise = false; % set all noise to 0


opts.simName = ''; %can be specified to save the simulation output into its own directory
