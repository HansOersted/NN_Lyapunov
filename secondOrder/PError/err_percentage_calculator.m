clear
close all

%% Define A  (lambda, n1)
% (2, 100)
% A = [ 1.0001  -0.0001;
%      -0.0001   1.0001 ];

% (2, 500)
% A = [ 1.0116  -0.0109;
%      -0.0109   1.0103 ];

% (2, 2000)
% A = [ 1.0151  -0.0141;
%      -0.0141   1.0151 ];

% random example
A = [ 1    0;
      0    1 ];

% (0.1, 1)
% A = [ 3.6724   -2.1600;
%      -2.1600    2.8910 ];

% (0.1, 10)
% A = [ 1.3343   -0.1308;
%      -0.1308    1.0533 ];

% (0.1, 100)
% A = [ 1.7746    0.2165;
%       0.2165    1.1135 ];

% (0.1, 200)
% A = [ 1.0130   -0.0105;
%      -0.0105    1.0085 ];

% (0.1, 500)
% A = [ 1.8340   -0.6995;
%      -0.6995    1.5895 ];

% (0.1, 2000)
% A = [ 4.3269   -2.9712;
%      -2.9712    3.6536 ];

%% load the candidate trajectory for the test and count the points violating the constraint

lambda_val = 0.1;
count = 0;

dimension = 2; % dimension of the state
length = 100;
sample_time = 0.05;

simulation_time = length * sample_time;
exploring_sample = [];

n2 = 1;  % number of samples for exploring

exploring_sample = [];
derivative_exploring_sample = [];
h_history = [];

for i = 1 : n2
    initial(i).init_dq = [0; 0];
    initial(i).init_q = [0.8/3; 0.5/3];
    init_dq = initial(i).init_dq;
    init_q = initial(i).init_q;

    data_from_simulink = sim('data');
    
    q_values = data_from_simulink.q.signals.values;
    dq_values = data_from_simulink.dq.signals.values;
    ddq_values = data_from_simulink.ddq.signals.values;

    de_values = data_from_simulink.de.signals.values;
    dde_values = -ddq_values;
    
    q_vec = squeeze(q_values)';
    dq_vec = squeeze(dq_values)';
    ddq_vec = squeeze(ddq_values)';
    
    de_vec = squeeze(de_values)';
    dde_vec = squeeze(dde_values)';

    derivative_training_sample(i).data = de_vec;
    derivative_derivative_training_sample(i).data = dde_vec;

    for idx = 1 : length
        % Extract current time step data
        de = derivative_training_sample(i).data(idx, :)'; % Tracking error derivative
        de1 = de(1);
        de2 = de(2);

        dde = derivative_derivative_training_sample(i).data(idx, :)'; % Tracking error second derivative
        dde1 = dde(1);
        dde2 = dde(2);

        h = [ 1/2 * (de1 ^ 2) ^ (-3/4) * de1 * dde1   1/2 * (de2 ^ 2) ^ (-3/4) * de2 * dde2 ] * A * [ (de1 ^ 2) ^ (1/4) ; (de2 ^ 2) ^ (1/4) ] ...
                            + [ (de1 ^ 2) ^ (1/4)  (de2 ^ 2) ^ (1/4) ] * A * [ 1/2 * (de1 ^ 2) ^ (-3/4) * de1 * dde1 ; 1/2 * (de2 ^ 2) ^ (-3/4) * de2 * dde2 ] ... 
                            + lambda_val * [ (de1 ^ 2) ^ (1/4)  (de2 ^ 2) ^ (1/4) ] * A * [ (de1 ^ 2) ^ (1/4) ; (de2 ^ 2) ^ (1/4) ];
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