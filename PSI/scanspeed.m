function scanspeed

    %There's a maximum scan speed that a linear galvo can attain
    %This is ~500Hz
    
    %If we use linear galvos, we are then forced to use multiple pulses per
    %pixel, and lower peak powers, than we might otherwise want to. This in
    %turn raises our average power (if we use 2 pulses/pixel, we
    %essentially double our rep rate on the photobleaching/avg power
    %curve)
    
    %We may be able to run linear galvos in some kind of 'open loop'
    %configuration
    
    %If we use resonant galvos, we can run these at a single higher speed,
    %~4kHz, but they will be hard to synchronize?
    
    %We can also use 1D AODs as scanners; downside is lower transmission (~70%);
    
    
    
    resolution = 1000;
    
    reprates = [0.5 1 2.5 5 10]*1E6;
    
    linerates = reprates/(4*resolution);
    
    figure, plot(reprates, linerates)
    
end