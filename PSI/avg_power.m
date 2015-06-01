function avg_power

pow_05 = 30; %laser power required for a conventional 2P to achieve 0.5 photons per pixel per pulse at 1.1 NA, in mW

%display the average power required
brightnesses = [0 0.01 0.05 0.1 0.5 1 2.5 5 10 15 20 30 40 60 80 100]; %photons per pulse per pixel
query_res = 700; %we'll calculate the power required for this resolution per axis
damage_coeff = 2.5; %the nonlinear photodamage power law is probably somewhere between 2 and 3
pow_per_pixel = (pow_05/80)*sqrt((brightnesses/0.5));  %Assume: 10mW at 80MHz produces 0.5 photons per pixel per pulse
pow_per_pulse = query_res/2 * pow_per_pixel; %this is the average power per MHz of rep rate
rep_rates = [0.5 1 2.5 5 10 20];

bleach_pps = 1200; %pulses per second for an 80MHz conventional 2P that induce photodamage at this power. This will be smaller for smaller pixel sizes!


pow = rep_rates'*pow_per_pulse ./1000;

colors = parula(length(rep_rates));

%Compute brightness vs avg power and photodamage for different rep rates... 
% For imaging a single plane:
figure('name', 'Single Plane', 'color', 'w');
L = cell(1,length(rep_rates));
for RR = 1:length(rep_rates)
    FPS = (rep_rates(RR)*1E6)/(4*query_res);
    plot(4*brightnesses*FPS, pow(RR,:), 'color', colors(RR,:), 'linewidth', 2) %brightness multiplied by 4 because 4 lines; this is the photon count per second
    hold on
    
    L{2*(RR-1)+1} = [int2str(FPS) ' frames/s (' num2str(rep_rates(RR)) ' MHz)']; %query_res multiplied by 4 because 4 lines
    L{2*(RR-1)+2} = [''];
    
    b_thresh(RR) = ((bleach_pps/(4*FPS)).^(1/damage_coeff)).^2 * 0.5*4*FPS; %brightness threshold
    plot([b_thresh(RR) b_thresh(RR)], [0 1], 'color', colors(RR,:), 'linestyle', '--', 'linewidth', 2)
    
    %pd_rate = (pow_per_pixel/(10/80)).^damage_coeff * (4 * FPS)/3200;  %3200 is an estimate of the number of laser pulses per second a pixel receives at damage threshold at 10mW
    %plot(ax2, 4*brightnesses*FPS, pd_rate, 'color', colors(RR,:), 'linewidth', 2, 'linestyle', ':')
    %hold(ax2, 'on')
end
plot([0 5000], [0.4 0.4], 'color', 'r', 'linewidth', 1.5)
set(gca, 'ylim', [0 1], 'xlim', [0 2500], 'tickdir', 'out', 'linewidth', 2, 'box', 'off')
xlabel('Brightness (#photons/pixel/second)')
ylabel('Average Power (W)')
title(['Imaging a single plane at ' int2str(query_res) ' x ' int2str(query_res) ' pixels'])
h = legend(L);


%For imaging a volume of 20 planes
figure('name', '20 planes', 'color', 'w');
L = cell(1,length(rep_rates));
for RR = 1:length(rep_rates)
    FPS = (rep_rates(RR)*1E6)/(20*4*query_res); %volumes per second
    plot(4*brightnesses*FPS, pow(RR,:), 'color', colors(RR,:), 'linewidth', 2) %brightness multiplied by 4 because 4 lines; this is the photon count per second
    hold on
    
    L{2*(RR-1)+1} = [num2str(FPS) ' volumes/s (' num2str(rep_rates(RR)) ' MHz)']; %query_res multiplied by 4 because 4 lines
    L{2*(RR-1)+2} = [''];
    
    b_thresh(RR) = ((bleach_pps/(4*FPS)).^(1/damage_coeff)).^2 * 0.5*4*FPS; %brightness threshold
    plot([b_thresh(RR) b_thresh(RR)], [0 1], 'color', colors(RR,:), 'linestyle', '--', 'linewidth', 2)
    
    %pd_rate = (pow_per_pixel/(10/80)).^damage_coeff * (4 * FPS)/3200;  %3200 is an estimate of the number of laser pulses per second a pixel receives at damage threshold at 10mW
    %plot(ax2, 4*brightnesses*FPS, pd_rate, 'color', colors(RR,:), 'linewidth', 2, 'linestyle', ':')
    %hold(ax2, 'on')
end
plot([0 5000], [0.4 0.4], 'color', 'r', 'linewidth', 1.5)
set(gca, 'ylim', [0 1], 'xlim', [0 1000], 'tickdir', 'out', 'linewidth', 2, 'box', 'off')
xlabel('Brightness (average #photons/pixel/second)')
ylabel('Average Power (W)')
title(['Imaging a 20-plane volume at ' int2str(query_res) ' x ' int2str(query_res) ' pixels'])
h = legend(L);

