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
    fprintf('Current acceptance rate = %.4f\r', i / j);  % Print acceptance rate
end

fprintf('\nFinal acceptance rate = %.4f\n', N / j);

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






