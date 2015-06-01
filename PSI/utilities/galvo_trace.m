function galvo_trace (duty_cycle, res)

%calculates a minimum-acceleration galvo trace with a one-directional sweep

%  0<duty cycle<1
%res is the number of discrete samples to produce

f = (1-duty_cycle)/duty_cycle;
a = (1+f)/(f.^2);

x_i = -1/(2*a);
b = 1 + a*(x_i.^2); 

x_eval = linspace(x_i,x_i+f, res*(f/(2*(f+1))) +1);
parabola = -a*x_eval.^2 + b;

Y = [-fliplr(parabola(2:end)) linspace(-1,1, res/(f+1)+1) parabola(2:end-1)];
figure, plot(repmat(Y, 1, 10))
figure, plot(diff(Y,2))
disp('Acceleration:')
max(diff(Y,2))*res
disp('Mirror range as multiple of linear scan range:')
(max(Y)-min(Y))/2