
%%  There are 2 types of the samples.
%   n1 samples: number of samples for training
%   n2 samples: number of samples for exploring

% sample_time = 0.04;  % unless you wanna do the numerical calculation
dimension = 1; % dimension of the state
% length = 100;

% n1 = 100; % number of samples for training
% n2 = 10;  % number of samples for exploring

% for i = 1 : n1
%     training_sample(i).data = zeros(length, dimension);
%     derivative_training_sample(i).data = zeros(length, dimension);
% end
% 
% for i = 1 : n2
%     exploring_sample(i).data = zeros(length, dimension);
%     derivative_exploring_sample(i).data = zeros(length, dimension);
% end

%% Prepare for training

gamma = 0.01;
h = 32; % width of the hidden layer
learning_rate = 1e-3;
num_epochs = 3;

% Define the weight for the NN
L1 = randn(h, dimension); % from input to hidden layer 1
b1 = zeros(h, 1);

L2 = randn(h, h); % from hidden layer 1 to hidden layer 2
b2 = zeros(h, 1);

L_out = randn(dimension * (2 * dimension), h); % from hidden layer to output
b_out = zeros(dimension * (2 * dimension), 1);

%% Fix lambda
lambda_val = 4; % Fixed lambda value

%%  Training loop
%   grad(V) * dx + lambda * V <= - gamma
%     <==>
%   (2 * (L*L' + I) * x)' * dx + lambda * x' * (L*L' + I) * x <= - gamma
%     <==>
%   2 * x' * (L*L' + I) * dx + lambda * x' * (L*L' + I) * x <= - gamma
%   der(h)/der(L) = grad_constraint = 4 * x * dx' * L_pred + 2 * lambda_val * x * x' * L_pred

%% Training loop
loss_history = zeros(num_epochs, 1);
A_history = zeros(num_epochs, 1);
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
            x = training_sample(i).data(t, :)'; % Current state (column vector)
            dx = derivative_training_sample(i).data(t, :)'; % State derivative
            
            % Forward pass
            hidden1 = tanh(L1 * x + b1);
            hidden2 = tanh(L2 * hidden1 + b2);
            L_pred = reshape(L_out * hidden2 + b_out, dimension, 2 * dimension);
            
            % Constraint computation
            A = L_pred * L_pred' + eye(dimension); % Coefficient matrix
            A_history_each_epoch = A;
            constraint = 2 * (x' * A * dx) + lambda_val * (x' * A * x) + gamma; % Use fixed lambda_val
            constraint_history_each_epoch = constraint;
            
            % Loss computation
            constraint_violation = max(0, constraint); % constraint_violation should be non-positive
            % Total loss
            loss = constraint_violation;
            total_loss = total_loss + loss;

            % Gradient computation
            if constraint_violation > 0
                grad_constraint = 4 * x * dx' * L_pred + 2 * lambda_val * x * x' * L_pred; % der(h)/der(L), h: constraint
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
            dL1 = dL1 + grad_hidden1 * x';
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
    A_history(epoch) = A_history_each_epoch;
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
plot(A_history, 'LineWidth', 2);
xlabel('Epoch');
ylabel('A');
title('A History');
grid on;

figure;
plot(constraint_history, 'LineWidth', 2);
xlabel('Epoch');
ylabel('Constraint');
title('Constraint History');
grid on;