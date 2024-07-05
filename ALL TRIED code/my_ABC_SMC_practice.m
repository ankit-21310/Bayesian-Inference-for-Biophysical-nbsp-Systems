
clear;
close all;
clc;

% % % % rng(42); % Set the random seed for reproducibility
% % % % 
% % % % % Lotka-Volterra ODE system
% % % % lv_ode = @(t, y, a, b) [a * y(1) - y(1) * y(2); b * y(1) * y(2) - y(2)];
% % % % 
% % % % % ABC-SMC parameters
% % % % T = 3; % Number of generations
% % % % N = 100; % Number of particles in each generation
% % % % epsilon = [30.0, 16.0, 6.0, 5.0, 4.3]; % Tolerance levels for each generation
% % % % prior_min = -10; % Prior distribution boundaries
% % % % prior_max = 10; % Uniform prior [-10, 10]
% % % % perturb_variance = 0.1; % Perturbation variance for random walk
% % % % 
% % % % % Observed data from the Lotka-Volterra model
% % % % a = 1; % True parameter 'a'
% % % % b = 1; % True parameter 'b'
% % % % x0 = [1.0; 0.5]; % Initial conditions x(0)=1 and y(0)=0.5
% % % % tspan = [0, 15]; % Time span for solving the ODE
% % % % 
% % % % % Solve the deterministic Lotka-Volterra system
% % % % [t_, Y] = ode15s(@(t_, Y) lv_ode(t_, Y, a, b), tspan, x0);
% % % % 
% % % % % Sample data points from the trajectories and add Gaussian noise
% % % % sample_times = linspace(0, 15, 8); % Times to sample data
% % % % prey_data = interp1(t_, Y(:, 1), sample_times) + randn(1, 8) * 0.5; % Noisy prey data
% % % % predator_data = interp1(t_, Y(:, 2), sample_times) + randn(1, 8) * 0.5; % Noisy predator data
% % % % 
% % % % % Our observed data x_d, y_d are:
% % % % x_d = prey_data;
% % % % y_d = predator_data;
% % % % 
% % % % % Distance function to calculate squared differences
% % % % calc_distance = @(x, y, x_d, y_d) sum((x - x_d).^2) + sum((y - y_d).^2);
% % % % 
% % % % % Particles and weights
% % % % 
% % % % data_generation_steps = zeros(1,T);
% % % % weights = zeros(T, N); % Weights for each population
% % % % pop_thetas = zeros(T, N, 2); % Particles (each element is a 2-element vector)
% % % % 
% % % % % ABC-SMC algorithm
% % % % for t = 1:T
% % % % 
% % % %     i = 1;
% % % %     j = 0; % data generation step in population t
% % % %     while i<= N
% % % %         x = zeros(1,8);
% % % %         y = zeros(1,8);
% % % %         theta_s_s = [0,0];
% % % % 
% % % %         if t == 1
% % % %             % directly sample theta_s_s from prior
% % % %             theta_s_s = [unifrnd(prior_min, prior_max), unifrnd(prior_min, prior_max)];
% % % % 
% % % %         else
% % % %             % sample theta_s from previous population
% % % %             [max_weight, max_index] = max(weights(t-1, :)); % Returns the max weight and its index
% % % % 
% % % %             theta_s = pop_thetas(t-1,max_index, :);  % corresponding theta from previous population
% % % % 
% % % %             perturbation = perturb_variance * unifrnd(-1, 1, [1, 2]);
% % % % 
% % % %             % theta_s_s kernal Kt(theta/theta_s)
% % % %             theta_s_s = theta_s + perturbation ; %perturb the particle   ?????????
% % % % 
% % % %         end
% % % % 
% % % %          % simulate dataset X_star from the model 
% % % %          [t_ , Y] = ode15s(@(t_ , Y) lv_ode(t_, Y, theta_s_s(1), theta_s_s(2)), tspan, x0);
% % % %          simulated_prey_data = interp1(t_, Y(:, 1), sample_times); % Prey data x
% % % %          simulated_predator_data = interp1(t_, Y(:, 2), sample_times); % Predator data y
% % % % 
% % % %          x = simulated_prey_data;
% % % %          y = simulated_predator_data;
% % % %          j = j + 1;
% % % % 
% % % %          %fprintf('Current j  = %i\r',  j);
% % % % 
% % % %          if ( theta_s(1) == 0 && theta_s(2) == 0 ) || theta_s(1) == 0 || theta_s(2) == 0
% % % %             % calculate distance d(X_star,Xo)
% % % %             distance = calc_distance(x, y, x_d, y_d);
% % % %             if distance < epsilon(t)
% % % % 
% % % %                 % accept the particle theta_s_s store in the res
% % % %                 pop_thetas(t,i, :) = theta_s_s;
% % % % 
% % % %                 % calculate weight of the particle wi of t
% % % %                 if t == 1
% % % %                     weights(t, i) = 1;
% % % %                 else
% % % %                     numerator = unifrnd(prior_min, prior_max);
% % % %                     deominator = 0;
% % % %                     for k = 1:N
% % % %                         deominator = deominator + weights(t-1,k)*unifrnd(-1, 1)*perturb_variance ; 
% % % %                     end 
% % % % 
% % % %                     weights(t, i) = numerator/deominator;
% % % %                 end
% % % % 
% % % %                 % increment i 
% % % %                 i = i +1;
% % % %             end
% % % %          end
% % % %     end 
% % % % 
% % % %     % store the data generation steps j  
% % % %     data_generation_steps(t) = j;
% % % % 
% % % % 
% % % %     % normalize the weight wt
% % % %     row_sum = sum(weights(t, :)); % Calculate the row's total sum
% % % %     weights(t, :) = weights(t, :) / row_sum;
% % % % 
% % % % 
% % % % end




clear
close all
clc

% Set the random seed for reproducibility
rng(42); % Can choose any fixed number for the seed

% Define the Lotka-Volterra system with parameters a and b
% dx/dt = a * y(1) - y(1) * y(2)
% dy/dt = b * y(1) * y(2) - y(2)
lv_ode = @(t, y, a, b) [a * y(1) - y(1) * y(2); b * y(1) * y(2) - y(2)];

% Parameters for solving the LV system
a = 1; % Parameter 'a'
b = 1; % Parameter 'b'
x0 = [1.0; 0.5]; % Initial conditions x(0)=1 and y(0)=0.5
tspan = [0, 15]; % Time span for solving the ODE

% Solve the ODE for the Lotka-Volterra system using ode15s
[t, Y] = ode15s(@(t, Y) lv_ode(t, Y, a, b), tspan, x0);

% Sample data points from the trajectories and add Gaussian noise
sample_times = linspace(0, 15, 8); % Times to sample data
prey_data = interp1(t, Y(:, 1), sample_times) + randn(1, 8) * 0.5; % Prey data with noise
predator_data = interp1(t, Y(:, 2), sample_times) + randn(1, 8) * 0.5; % Predator data with noise

% Our observed data x_d, y_d are:
x_d = prey_data;
y_d = predator_data;

% Plot the deterministic trajectories for prey and predator
figure;
plot(t, Y(:, 1), 'k-', 'LineWidth', 1.5); % Prey trajectory (solid curve)
hold on;
plot(t, Y(:, 2), 'k--', 'LineWidth', 1.5); % Predator trajectory (dashed curve)

% Add noisy data points for prey and predator
scatter(sample_times, prey_data, 'ko', 'filled'); % Prey data (circles)
scatter(sample_times, predator_data, 'k^', 'filled'); % Predator data (triangles)

% Customize the plot with a y-axis limit
ylim([0, 4]); % Set y-axis limit from 0 to 4

% Additional plot customizations
xlabel('Time');
ylabel('Population');
legend('Prey (Deterministic)', 'Predator (Deterministic)', 'Prey Data', 'Predator Data', 'Location', 'Best');
title('Lotka-Volterra Trajectories and Noisy Data Points');
grid on;

hold off;

% ================= Now ABC Rejection Sampler =========================

N = 1000;  % Number of accepted particles
epsilon = 5;  % Constant tolerance level
prior_min = -10;  % Uniform[-10, 10] prior
prior_max = 10;  

observed_data = {prey_data, predator_data}; % Data with noise N(1, 0.5^2)

res = zeros(N, 3);  % Results matrix
i = 1;  % Counter for accepted particles
j = 1;  % Counter for total proposals

tic;
while i <= N
    % Sample from the prior
    
    theta_star = [unifrnd(prior_min, prior_max), unifrnd(prior_min, prior_max)];

    % Simulate dataset from the model with ode15s
    [t, Y] = ode15s(@(t, Y) lv_ode(t, Y, theta_star(1), theta_star(2)), tspan, x0);

    simulated_prey_data = interp1(t, Y(:, 1), sample_times); % Prey data x
    simulated_predator_data = interp1(t, Y(:, 2), sample_times); % Predator data y

    % Calculate distance
    distance = calc_distance(simulated_prey_data, simulated_predator_data, x_d, y_d);

    if distance <= epsilon
        % Store results
        res(i, :) = [theta_star(1), theta_star(2), distance];
        i = i + 1; % Update accepted particle counter
    end

    j = j + 1;  % Update total proposal counter
    
    fprintf('j = %.4f\n',  j);
    % fprintf('Current acceptance rate = %.4f\r', i / j);  % Print acceptance rate
end

fprintf('\nFinal acceptance rate = %.7f\n', N / j);
fprintf('j = %.4f\n',  j);

toc;
% Visualization of the results
figure;

% Subplot for the histogram of 'a'
subplot(1, 2, 1); 
histogram(res(:, 1)); % Histogram of the first column ('a' data)
xlabel('a'); % Label for x-axis
ylabel('Frequency'); % Label for y-axis
title('Histogram of Parameter a'); % Title for the subplot

% Subplot for the histogram of 'b'
subplot(1, 2, 2); 
histogram(res(:, 2)); % Histogram of the second column ('b' data)
xlabel('b'); % Label for x-axis
ylabel('Frequency'); % Title for y-axis
title('Histogram of Parameter b');

hold off;

% Distance function to calculate squared differences
function distance = calc_distance(x, y, x_d, y_d)
    x_squared_diff = (x - x_d).^2; % Squared differences for prey
    y_squared_diff = (y - y_d).^2; % Squared differences for predator
    distance = sum(x_squared_diff) + sum(y_squared_diff); % Total distance
end
















































% % % % % 
% % % % % lv_ode = @(t, y, a, b) [a * y(1) - y(1) * y(2); b * y(1) * y(2) - y(2)];
% % % % % sample_times = linspace(0, 15, 8);
% % % % % tic;
% % % % %  % Sample from the prior
% % % % %     theta_star = [unifrnd(-10, 10), unifrnd(-10, 10)];
% % % % % 
% % % % %     % Simulate dataset from the model with ode15s
% % % % %     [t, Y] = ode15s(@(t, Y) lv_ode(t, Y, theta_star(1), theta_star(2)), [0,15], [1,0.5]);
% % % % % 
% % % % %     simulated_prey_data = interp1(t, Y(:, 1), sample_times); % Prey data x
% % % % %     simulated_predator_data = interp1(t, Y(:, 2), sample_times); % Predator data y
% % % % % 
% % % % %     % Calculate distance
% % % % %     % distance = calc_distance(simulated_prey_data, simulated_predator_data, x_d, y_d);
% % % % % 
% % % % % 
% % % % % 
% % % % % toc;


% 
% for i = 1:1000
%     a = unifrnd(-10, 10);
%     fprintf('a = %.2f\n',  a);
% end






% % 
% % T = 5; % Number of time steps
% % N = 1000; % Number of particles
% % weights = zeros(T, N); % Initialize weights
% % pop_thetas = zeros(T, N, 2); % Initialize particles with [a, b]
% % 
% % % Generate random example data
% % for t = 1:T
% %     pop_thetas(t, :, 1) = randn(1, N) * 2 + 5; % 'a' values (Gaussian with mean 5)
% %     pop_thetas(t, :, 2) = randn(1, N) * 3 + 10; % 'b' values (Gaussian with mean 10)
% % end
% % 
% % % Create a single figure for all time steps with a taller aspect ratio
% % figure('Position', [100, 100, 700, 1000]); % Set figure size (wider and taller)
% % for t = 1:T
% %     % Left subplot for 'a' values
% %     subplot(T, 2, 2 * (t - 1) + 1); % `T` rows, 2 columns, left column for 'a'
% %     histogram(pop_thetas(t, :, 1), 10); % 30 bins for histogram
% %     title(['Histogram of a for t = ', num2str(t)]);
% %     xlabel('a');
% %     ylabel('Frequency');
% %     ylim([0, 400]); % Set y-axis limit to [0, 400]
% %     yticks(0:100:400); % Set y-axis ticks at intervals of 100
% % 
% %     % Right subplot for 'b' values
% %     subplot(T, 2, 2 * t); % `T` rows, 2 columns, right column for 'b'
% %     histogram(pop_thetas(t, :, 2), 10); % 30 bins for histogram
% %     title(['Histogram of b for t = ', num2str(t)]);
% %     xlabel('b');
% %     ylabel('Frequency');
% %     ylim([0, 400]); % Set y-axis limit to [0, 400]
% %     yticks(0:100:400); % Set y-axis ticks at intervals of 100
% % end
