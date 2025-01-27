%   dx = -x
%   x = x_0 * exp(-t)
%   dx = -x = -x_0 * exp(-t)
%   a good Lyapunov candidate: V = x^2 ==> dV = 2 * x * dx = -2 * x^2

clear
close all

%%  

sample_time = 0.05;
length = 100;

x_0 = 1;

t = 0 : sample_time : (length - 1) * sample_time;
plot(t, x_0 * exp(-t));
hold on
plot(t, -x_0 * exp(-t));

%%

dimension = 1; % dimension of the state


n1 = 200; % number of samples for training
n2 = 10;  % number of samples for exploring

for i = 1 : n1
    training_sample(i).data = zeros(length, dimension);
    derivative_training_sample(i).data = zeros(length, dimension);
end

for i = 1 : n1
    for idx = 1 : length
        training_sample(i).data(idx) = x_0 * exp(-(idx - 1) * sample_time);
        derivative_training_sample(i).data(idx) = -x_0 * exp(-(idx - 1));
    end
end


for i = 1 : n2
    exploring_sample(i).data = zeros(length, dimension);
    derivative_exploring_sample(i).data = zeros(length, dimension);
end