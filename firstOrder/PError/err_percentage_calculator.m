clear
close all

%% Define A
A = 1;

%% load the candidate trajectory for the test and count the points violating the constraint

lambda_val = 1;
count = 0;

dimension = 1; % dimension of the state
length = 100;
sample_time = 0.05;

n2 = 1;  % number of samples for exploring

exploring_sample = [];
derivative_exploring_sample = [];
h_history = [];

for i = 1 : n2
    x_0 = 1;
    for idx = 1 : length
        exploring_sample(idx) = x_0 * exp(-(idx - 1) * sample_time);
        derivative_exploring_sample(idx) = -x_0 * exp(-(idx - 1) * sample_time);
        x = exploring_sample(idx);
        dx = derivative_exploring_sample(idx);
        h = 2 * (x' * A * dx) + lambda_val * (x' * A * x);
        h_history = [h_history; h];
        if h > 0
            count = count + 1;
        end
    end
end

count

%% Calculate Bh

max_Bh = 0;

for idx = 1 : length
   if abs(h_history(idx)) > max_Bh
      max_Bh = abs(h_history(idx));
   end
end

%% Calculate err_percentage

err_percentage = count/length


%% upper bound
n1 = 100;
syms K1 K2

gamma = 0.01;
delta = 0.01;
Bh = max_Bh;

upper_bound = K1 * (log10(n1))^3 / (gamma ^ 2 * n1) + K2 * log10(log10(4*Bh/gamma)/delta)/n1;