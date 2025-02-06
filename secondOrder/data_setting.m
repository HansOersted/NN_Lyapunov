clear
close all

%% load data

sample_time = 0.05;
length = 100;
simulation_time = sample_time * length;

n1 = 1;
dimension = 2;

for i = 1 : n1
    initial(i).init_dq = [0; 0];
    initial(i).init_q = [0.8/i; 0.5/i];
end

for i = 1 : n1
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
end

%% Prepare for training

gamma = 0.01;
h = 32; % width of the hidden layer
learning_rate = 1e-3;
num_epochs = 1000;

% Define the weight for the NN
L1 = randn(h, dimension); % from input to hidden layer 1
b1 = zeros(h, 1);

L2 = randn(h, h); % from hidden layer 1 to hidden layer 2
b2 = zeros(h, 1);

L_out = randn(dimension * (2 * dimension), h); % from hidden layer to output
b_out = zeros(dimension * (2 * dimension), 1);

%% Fix lambda
lambda_val = 0.1; % Fixed lambda value

%%  Training loop

%   V = [ (de1 ^ 2) ^ (1/4)  (de2 ^ 2) ^ (1/4) ] * (L*L' + I) * [ (de1 ^ 2) ^ (1/4) ; (de2 ^ 2) ^ (1/4) ] 

%   grad(V) * dx + lambda * V <= - gamma
%     <==>
%   [ 1/2 * (de1 ^ 2) ^ (-3/4) * de1 * dde1   1/2 * (de2 ^ 2) ^ (-3/4) * de2 * dde2 ] * A * [ (de1 ^ 2) ^ (1/4) ; (de2 ^ 2) ^ (1/4) ] 
%   + [ (de1 ^ 2) ^ (1/4)  (de2 ^ 2) ^ (1/4) ] * A * [ 1/2 * (de1 ^ 2) ^ (-3/4) * de1 * dde1 ; 1/2 * (de2 ^ 2) ^ (-3/4) * de2 * dde2 ] 
%   + lambda * V <= - gamma

%   der(h)/der(L) = grad_constraint = 
%

%   Note that the derivative of  A * L * L' * B  is  ( (A * B).' + B * A ) * L

%% Training loop
loss_history = zeros(num_epochs, 1);
A_history_each_epoch = [];
constraint_history = zeros(num_epochs, 1);
constraint_history_each_epoch = [];

for epoch = 1 : num_epochs
    total_loss = 0;
    currentEpochs = epoch

    % Initialize gradients
    dL1 = zeros(size(L1));
    db1 = zeros(size(b1));
    dL2 = zeros(size(L2));
    db2 = zeros(size(b2));
    dL_out = zeros(size(L_out));
    db_out = zeros(size(b_out));

    for i = 1 : n1
        for t = 1 : length
            % Extract current time step data
            de = derivative_training_sample(i).data(t, :)'; % Tracking error derivative
            de1 = de(1);
            de2 = de(2);

            dde = derivative_derivative_training_sample(i).data(t, :)'; % Tracking error second derivative
            dde1 = dde(1);
            dde2 = dde(2);

            sqrt_abs_de = sqrt(abs(de));
            
            % Forward pass
            hidden1 = tanh(L1 * sqrt_abs_de + b1);
            hidden2 = tanh(L2 * hidden1 + b2);
            L_pred = reshape(L_out * hidden2 + b_out, dimension, 2 * dimension);
            
            % Constraint computation
            A = L_pred * L_pred' + eye(dimension); % Coefficient matrix
            A_history_each_epoch = A;
            constraint = [ 1/2 * (de1 ^ 2) ^ (-3/4) * de1 * dde1   1/2 * (de2 ^ 2) ^ (-3/4) * de2 * dde2 ] * A * [ (de1 ^ 2) ^ (1/4) ; (de2 ^ 2) ^ (1/4) ] ...
                            + [ (de1 ^ 2) ^ (1/4)  (de2 ^ 2) ^ (1/4) ] * A * [ 1/2 * (de1 ^ 2) ^ (-3/4) * de1 * dde1 ; 1/2 * (de2 ^ 2) ^ (-3/4) * de2 * dde2 ] ... 
                            + lambda_val * [ (de1 ^ 2) ^ (1/4)  (de2 ^ 2) ^ (1/4) ] * A * [ (de1 ^ 2) ^ (1/4) ; (de2 ^ 2) ^ (1/4) ] + gamma; % Use fixed lambda_val 
            constraint_history_each_epoch = constraint;
            
            % Loss computation
            constraint_violation = max(0, constraint); % constraint_violation should be non-positive
            % Total loss
            loss = constraint_violation;
            total_loss = total_loss + loss;

            % Gradient computation
% [ 1/2 * (de1 ^ 2) ^ (-3/4) * de1 * dde1   1/2 * (de2 ^ 2) ^ (-3/4) * de2 * dde2 ] * A * [ (de1 ^ 2) ^ (1/4) ; (de2 ^ 2) ^ (1/4) ] ...
% + [ (de1 ^ 2) ^ (1/4)  (de2 ^ 2) ^ (1/4) ] * A * [ 1/2 * (de1 ^ 2) ^ (-3/4) * de1 * dde1 ; 1/2 * (de2 ^ 2) ^ (-3/4) * de2 * dde2 ] ... 
% + lambda_val * [ (de1 ^ 2) ^ (1/4)  (de2 ^ 2) ^ (1/4) ] * A * [ (de1 ^ 2) ^ (1/4) ; (de2 ^ 2) ^ (1/4) ] + gamma;
            if constraint_violation > 0
                A1 = [ 1/2 * (de1 ^ 2) ^ (-3/4) * de1 * dde1   1/2 * (de2 ^ 2) ^ (-3/4) * de2 * dde2 ];
                B1 = [ (de1 ^ 2) ^ (1/4) ; (de2 ^ 2) ^ (1/4) ];
                A2 = [ (de1 ^ 2) ^ (1/4)  (de2 ^ 2) ^ (1/4) ];
                B2 = [ 1/2 * (de1 ^ 2) ^ (-3/4) * de1 * dde1 ; 1/2 * (de2 ^ 2) ^ (-3/4) * de2 * dde2 ];
                A3 = [ (de1 ^ 2) ^ (1/4)  (de2 ^ 2) ^ (1/4) ];
                B3 = [ (de1 ^ 2) ^ (1/4) ; (de2 ^ 2) ^ (1/4) ];
                grad_constraint =   ( (A1 * B1).' + B1 * A1 ) * L_pred ...
                                  + ( (A2 * B2).' + B2 * A2 ) * L_pred ...
                                  + lambda_val * ( (A3 * B3).' + B3 * A3 ) * L_pred; 
                             % der(h)/der(L), h: constraint
            else
                grad_constraint = zeros(size(L_pred));
            end
            
            % Update gradients for weights
            grad_L_pred_reshaped = reshape(grad_constraint, size(L_pred));
            dL_out = dL_out + grad_L_pred_reshaped(:) * hidden2';
            db_out = db_out + grad_L_pred_reshaped(:);
            
            % Gradients for hidden layers
            grad_hidden2 = (L_out' * grad_L_pred_reshaped(:)) .* (1 - hidden2.^2); % tanh derivative
            dL2 = dL2 + grad_hidden2 * hidden1';
            db2 = db2 + grad_hidden2;
            
            grad_hidden1 = (L2' * grad_hidden2) .* (1 - hidden1.^2);
            dL1 = dL1 + grad_hidden1 * sqrt_abs_de';
            db1 = db1 + grad_hidden1;

            % Update weights
            L1 = L1 - learning_rate * dL1 / (n1 * length);
            b1 = b1 - learning_rate * db1 / (n1 * length);
            L2 = L2 - learning_rate * dL2 / (n1 * length);
            b2 = b2 - learning_rate * db2 / (n1 * length);
            L_out = L_out - learning_rate * dL_out / (n1 * length);
            b_out = b_out - learning_rate * db_out / (n1 * length);

        end
    end

    % Save loss history
    loss_history(epoch) = total_loss;
    A_history(epoch).history = A_history_each_epoch;
    constraint_history(epoch) = constraint_history_each_epoch;

    % % Print loss
    % if mod(epoch, 10) == 0
    %     fprintf('Epoch %d, Loss: %.4f\n', epoch, total_loss);
    % end
end

%% Plot results
figure;
plot(loss_history, 'LineWidth', 2);
xlabel('Epoch');
ylabel('Loss');
title('Training Loss');
grid on;

figure;
plot(constraint_history, 'LineWidth', 2);
xlabel('Epoch');
ylabel('Constraint');
title('Constraint History');
grid on;