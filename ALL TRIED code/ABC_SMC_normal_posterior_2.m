clear
close all
clc


% Set the random seed for reproducibility
rng(42); % You can choose any fixed number for the seed

% Define the Lotka-Volterra system with parameters a and b
% dx/dt = a * y(1) - y(1) * y(2)
% dy/dt = b * y(1) * y(2) - y(2)
lv_ode = @(t, y, a, b) [a * y(1) - y(1) * y(2); b * y(1) * y(2) - y(2)]; % dx/dt = a * y(1) - y(1) * y(2)

% Parameters for solving the LV system
a = 1; % Parameter 'a'
b = 1; % Parameter 'b'
x0 = [1.0; 0.5]; % Initial conditions x(0)=1 & y(0) = 0.5
tspan = [0, 15]; % Time span for solving the ODE

% Solve the ODE for the Lotka-Volterra system
[t, y] = ode45(@(t, y) lv_ode(t, y, a, b), tspan, x0);


% Sample data points from the trajectories and add Gaussian noise
sample_times = linspace(0, 15, 8); % Times to sample data
prey_data = interp1(t, y(:, 1), sample_times) + randn(1, 8) * 0.5; % Prey data with noise
predator_data = interp1(t, y(:, 2), sample_times) + randn(1, 8) * 0.5; % Predator data with noise

% our observed data x_d , y_d is :
x_d = prey_data;
y_d = predator_data;

% Plot the deterministic trajectories for prey and predator
figure;
plot(t, y(:, 1), 'k-', 'LineWidth', 1.5); % Prey trajectory (solid curve)
hold on;
plot(t, y(:, 2), 'k--', 'LineWidth', 1.5); % Predator trajectory (dashed curve)

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



% ================= Now ABC Rejection sampler =========================

N = 1000;  % number of accepted particles
epsilon = 5;  % constant tolerance level
prior_min = -10;  % U[-10,10] prior
prior_max = 10;  

observed_data = {prey_data, predator_data }; % with nois N(1,o.5^2)

res = zeros(N, 3);  % results matrix
i = 1;  % counter for accepted particles
j = 1;  % counter for total proposals



while i <= N
    % Sample from the prior
    % theta_star = [a*  b*]  

    theta_star = [unifrnd(prior_min, prior_max) unifrnd(prior_min, prior_max)];
    

    tic;
    % Simulate data set from the model
    [t, y] = ode45(@(t, y) lv_ode(t, y, theta_star(1), theta_star(2)), tspan, x0);
    sample_times = linspace(0, 15, 8); % Times to sample data
    simulated_prey_data = interp1(t, y(:, 1), sample_times) ; % Prey data x
    simulated_predator_data = interp1(t, y(:, 2), sample_times) ; % Predator data y
    
    toc;
  
    % Calculate distance
    distance = calc_distance(simulated_prey_data, simulated_predator_data,  x_d, y_d);
    
    if distance <= epsilon
        % Store results
        res(i, :) = [theta_star(1),theta_star(2), distance];
        i = i + 1;  % update accepted particle counter
    end
    j = j + 1;  % update total proposal counter
    fprintf('Current acceptance rate = %.4f\r', i / j);  % print acceptance rate
end



figure;

% Subplot for the histogram of 'a'
subplot(1, 2, 1); % 1 row, 2 columns, this is the first subplot
histogram(res(:, 1)); % Histogram of the first column ('a' data)
xlabel('a'); % Label for x-axis
ylabel('Frequency'); % Label for y-axis
title('Histogram of Parameter a'); % Title for the subplot

% Subplot for the histogram of 'b'
subplot(1, 2, 2); % 1 row, 2 columns, this is the second subplot
histogram(res(:, 2)); % Histogram of the second column ('b' data)
xlabel('b'); % Label for x-axis
ylabel('Frequency'); % Label for y-axis
title('Histogram of Parameter b'); % Title for the subplot




% Add a red dotted line at mu = 5
hold on; % Keep the current plot and add new plots
line([1 1], ylim, 'Color', 'r', 'LineStyle', '--'); % Draw a vertical line at x = 5
hold off; % Release the hold on the plot





function distance = calc_distance(x, y, x_d, y_d)
    
    % Calculate the squared differences for x and y
    x_squared_diff = (x - x_d).^2; % Squared differences for x
    y_squared_diff = (y - y_d).^2; % Squared differences for y
    
    % Calculate the total distance
    distance = sum(x_squared_diff) + sum(y_squared_diff);
end







% % 
% % % Define the Lotka-Volterra system with parameters a and b
% % lv_ode = @(t, y, a, b) [a * y(1) - b * y(1) * y(2); b * y(1) * y(2) - y(2)];
% % 
% % % Parameters for solving the LV system
% % a = 1; % Parameter 'a'
% % b = 1; % Parameter 'b'
% % x0 = [1.0; 0.5]; % Initial conditions
% % tspan = [0, 15]; % Time span for solving the ODE
% % 
% % % Solve the ODE for the Lotka-Volterra system
% % [t, y] = ode45(@(t, y) lv_ode(t, y, a, b), tspan, x0);
% % 
% % % Sample data points from the trajectories and add Gaussian noise
% % sample_times = linspace(0, 15, 8); % Times to sample data
% % prey_data = interp1(t, y(:, 1), sample_times) + randn(1, 8) * 0.5; % Prey data with noise
% % predator_data = interp1(t, y(:, 2), sample_times) + randn(1, 8) * 0.5; % Predator data with noise
% % 
% % % Plot the deterministic trajectories for prey and predator
% % figure;
% % plot(t, y(:, 1), 'k-', 'LineWidth', 1.5); % Prey trajectory (solid curve)
% % hold on;
% % plot(t, y(:, 2), 'k--', 'LineWidth', 1.5); % Predator trajectory (dashed curve)
% % 
% % % Overlay the data points for prey and predator
% % scatter(sample_times, prey_data, 70, 'ko', 'filled'); % Prey data (circles)
% % scatter(sample_times, predator_data, 70, 'k^', 'filled'); % Predator data (triangles)
% % 
% % % Customize the plot with a y-axis limit
% % ylim([0, 4]); % Set y-axis limit from 0 to 4
% % 
% % % Additional plot customizations
% % xlabel('Time');
% % ylabel('Population');
% % legend('Prey (Deterministic)', 'Predator (Deterministic)', 'Prey Data', 'Predator Data', 'Location', 'Best');
% % title('Lotka-Volterra Trajectories and Noisy Data Points');
% % grid on;
% % 
% % hold off;
% % 
% % 




% % % 
% % % % Define the Lotka-Volterra ODE system
% % % lv_ode = @(t, y, a, b) [a * y(1) - b * y(1) * y(2); b * y(1) * y(2) - y(2)];
% % % 
% % % % Parameters for the LV system
% % % a = 1; % Parameter 'a'
% % % b = 1; % Parameter 'b'
% % % x0 = [1.0; 0.5]; % Initial conditions for prey and predator
% % % tspan = [0, 15]; % Time span for solving the ODE
% % % 
% % % % Solve the ODE to get the deterministic solution
% % % [t, y] = ode45(@(t, y) lv_ode(t, y, a, b), tspan, x0);
% % % 
% % % % Define times to sample data and add Gaussian noise
% % % num_samples = 8; % Number of data points to sample
% % % sample_times = linspace(0, 15, num_samples); % Times to sample data
% % % noise_std = 0.5; % Standard deviation of Gaussian noise
% % % 
% % % % Sample data points and add noise
% % % x_sample = interp1(t, y(:, 1), sample_times); % Prey data without noise
% % % y_sample = interp1(t, y(:, 2), sample_times); % Predator data without noise
% % % 
% % % % Add Gaussian noise to sampled data points
% % % x_noisy = x_sample + randn(1, num_samples) * noise_std; % Prey data with noise
% % % y_noisy = y_sample + randn(1, num_samples) * noise_std; % Predator data with noise
% % % 
% % % % Plot the deterministic trajectories for prey and predator
% % % figure;
% % % plot(t, y(:, 1), 'k-', 'LineWidth', 1.5); % Prey trajectory (solid curve)
% % % hold on;
% % % plot(t, y(:, 2), 'k--', 'LineWidth', 1.5); % Predator trajectory (dashed curve)
% % % 
% % % % Add noisy data points to the plot
% % % scatter(sample_times, x_noisy, 70, 'ko', 'filled'); % Prey data (circles)
% % % scatter(sample_times, y_noisy, 70, 'k^', 'filled'); % Predator data (triangles)
% % % 
% % % % Set y-axis limit
% % % ylim([0, 4]); % Constrain y-axis to range 0 to 4
% % % 
% % % % Add plot labels, legend, and title
% % % xlabel('Time');
% % % ylabel('Population');
% % % legend('Prey (Deterministic)', 'Predator (Deterministic)', 'Prey Data', 'Predator Data', 'Location', 'Best');
% % % title('Trajectories of Prey and Predator with Noisy Data Points');
% % % grid on;
% % % 
% % % hold off;
% % % 

% % % 
% % % % Define the Lotka-Volterra ODE system
% % % lv_ode = @(t, y, a, b) [a * y(1) - b * y(1) * y(2); b * y(1) * y(2) - y(2)];
% % % 
% % % % Parameters for the LV system
% % % a = 1; % Parameter 'a'
% % % b = 1; % Parameter 'b'
% % % x0 = [1.0; 0.5]; % Initial conditions for prey and predator
% % % tspan = [0, 15]; % Time span for solving the ODE
% % % 
% % % % Solve the ODE to get the deterministic solution
% % % [t, y] = ode45(@(t, y) lv_ode(t, y, a, b), tspan, x0);
% % % 
% % % % Define times to sample data and add Gaussian noise
% % % num_samples = 8; % Number of data points to sample
% % % sample_times = linspace(0, 15, num_samples); % Times to sample data
% % % noise_std = 0.5; % Standard deviation of Gaussian noise
% % % 
% % % % Sample data points and add noise
% % % x_sample = interp1(t, y(:, 1), sample_times); % Prey data without noise
% % % y_sample = interp1(t, y(:, 2), sample_times); % Predator data without noise
% % % 
% % % % Add Gaussian noise to sampled data points
% % % x_noisy = x_sample + randn(1, num_samples) * noise_std; % Prey data with noise
% % % y_noisy = y_sample + randn(1, num_samples) * noise_std; % Predator data with noise
% % % 
% % % % Plot the deterministic trajectories for prey and predator
% % % figure;
% % % plot(t, y(:, 1), 'k-', 'LineWidth', 1.5); % Prey trajectory (solid curve)
% % % hold on;
% % % plot(t, y(:, 2), 'k--', 'LineWidth', 1.5); % Predator trajectory (dashed curve)
% % % 
% % % % Add noisy data points to the plot
% % % scatter(sample_times, x_noisy, 70, 'ko', 'filled'); % Prey data (circles)
% % % scatter(sample_times, y_noisy, 70, 'k^', 'filled'); % Predator data (triangles)
% % % 
% % % % Add plot labels, legend, and title
% % % xlabel('Time');
% % % ylabel('Population');
% % % legend('Prey (Deterministic)', 'Predator (Deterministic)', 'Prey Data', 'Predator Data', 'Location', 'Best');
% % % title('Trajectories of Prey and Predator with Noisy Data Points');
% % % grid on;
% % % 
% % % hold off;







% % % % % % % Define the Lotka-Volterra ODE system with parameters a and b
% % % % % % lv_ode = @(t, y, a, b) [a * y(1) - b * y(1) * y(2); b * y(1) * y(2) - y(2)];
% % % % % % 
% % % % % % % Initial conditions and parameter values for the deterministic LV system
% % % % % % x0 = [1.0; 0.5]; % Initial conditions for prey and predator
% % % % % % tspan = [0, 15]; % Time span for solving the ODE
% % % % % % a = 1; % Parameter 'a'
% % % % % % b = 1; % Parameter 'b'
% % % % % % 
% % % % % % % Solve the ODE to get the deterministic solution
% % % % % % [t, y] = ode45(@(t, y) lv_ode(t, y, a, b), tspan, x0);
% % % % % % 
% % % % % % % Sample eight data points from the solution and add Gaussian noise
% % % % % % num_samples = 8; % Number of data points to sample
% % % % % % sample_times = linspace(0, 15, num_samples); % Times to sample data
% % % % % % noise_std = 0.5; % Standard deviation of Gaussian noise
% % % % % % 
% % % % % % % Prey and predator data with Gaussian noise
% % % % % % x_sample = interp1(t, y(:, 1), sample_times); % Sampled prey data
% % % % % % y_sample = interp1(t, y(:, 2), sample_times); % Sampled predator data
% % % % % % x_data = x_sample + randn(1, num_samples) * noise_std; % Prey data with noise
% % % % % % y_data = y_sample + randn(1, num_samples) * noise_std; % Predator data with noise
% % % % % % 
% % % % % % % Create the plot showing deterministic trajectories and noisy data points
% % % % % % figure;
% % % % % % plot(t, y(:, 1), 'k-', 'LineWidth', 1.5); % Prey trajectory (solid curve)
% % % % % % hold on;
% % % % % % plot(t, y(:, 2), 'k--', 'LineWidth', 1.5); % Predator trajectory (dashed curve)
% % % % % % 
% % % % % % % Add noisy data points (circles for prey, triangles for predator)
% % % % % % scatter(sample_times, x_data, 70, 'ko', 'filled'); % Prey data (circles)
% % % % % % scatter(sample_times, y_data, 70, 'k^', 'filled'); % Predator data (triangles)
% % % % % % 
% % % % % % % Customize the plot with labels, legend, and grid
% % % % % % xlabel('Time');
% % % % % % ylabel('Population');
% % % % % % legend('Prey (Deterministic)', 'Predator (Deterministic)', 'Prey Data', 'Predator Data', 'Location', 'Best');
% % % % % % title('Trajectories of Prey and Predator with Noisy Data Points');
% % % % % % grid on;
% % % % % % 
% % % % % % hold off;
% % % % % % 

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % Define the Lotka-Volterra ODE system with parameters a and b
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % lv_ode = @(t, y, a, b) [a * y(1) - b * y(1) * y(2); b * y(1) * y(2) - y(2)];
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % Define initial conditions and parameter values
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % x0 = [1.0; 0.5]; % Initial conditions
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % tspan = [0, 15]; % Time span for solving the ODE
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % a = 1; % Parameter 'a'
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % b = 1; % Parameter 'b'
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % Solve the ODE to get the deterministic solution
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % [t, y] = ode45(@(t, y) lv_ode(t, y, a, b), tspan, x0);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % Define times to sample data and add Gaussian noise to create noisy data points
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % num_samples = 8; % Number of data points
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % sample_times = linspace(0, 15, num_samples); % Times to sample data
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % noise_std = 0.5; % Standard deviation of Gaussian noise
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % Sample prey and predator data from the deterministic solution
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % x_sample = interp1(t, y(:, 1), sample_times); % Prey samples
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % y_sample = interp1(t, y(:, 2), sample_times); % Predator samples
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % Add Gaussian noise to create noisy observations
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % x_data = x_sample + randn(1, num_samples) * noise_std; % Prey noisy data
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % y_data = y_sample + randn(1, num_samples) * noise_std; % Predator noisy data
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % Plot the deterministic trajectories for prey and predator
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % figure;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % plot(t, y(:, 1), 'k-', 'LineWidth', 1.5); % Prey (solid curve)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % hold on;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % plot(t, y(:, 2), 'k--', 'LineWidth', 1.5); % Predator (dashed curve)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % Overlay the noisy data points for prey and predator
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % scatter(sample_times, x_data, 70, 'ko', 'filled'); % Prey data (circles)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % scatter(sample_times, y_data, 70, 'k^', 'filled'); % Predator data (triangles)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % Customize the plot with labels, legend, title, and grid
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % xlabel('Time');
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % ylabel('Population');
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % legend('Prey (Deterministic)', 'Predator (Deterministic)', 'Prey Data', 'Predator Data', 'Location', 'Best');
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % title('Trajectories of Prey and Predator with Data Points');
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % grid on;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % hold off;







































% 
% % Define the Lotka-Volterra system with parameters a and b
% lv_ode = @(t, y, a, b) [a * y(1) - b * y(1) * y(2); b * y(1) * y(2) - y(2)];
% 
% % Parameters for solving the LV system
% a = 1; % Parameter a
% b = 1; % Parameter b
% x0 = [1.0; 0.5]; % Initial conditions
% tspan = [0, 15]; % Time span for solving the ODE
% 
% % Solve the ODE for the Lotka-Volterra system
% [t, y] = ode45(@(t, y) lv_ode(t, y, a, b), tspan, x0);
% 
% % Sample data points from the trajectories and add Gaussian noise
% sample_times = linspace(0, 15, 8); % Times to sample data
% prey_data = interp1(t, y(:, 1), sample_times) + randn(1, 8) * 0.5; % Prey data with noise
% predator_data = interp1(t, y(:, 2), sample_times) + randn(1, 8) * 0.5; % Predator data with noise
% 
% % Plot the deterministic trajectories for prey and predator
% figure;
% plot(t, y(:, 1), 'k-', 'LineWidth', 1.5); % Prey trajectory
% hold on;
% plot(t, y(:, 2), 'k--', 'LineWidth', 1.5); % Predator trajectory
% 
% % Overlay the data points for prey and predator
% scatter(sample_times, prey_data, 'ko', 'LineWidth', 1.5); % Prey data points (circles)
% scatter(sample_times, predator_data, 'k^', 'LineWidth', 1.5); % Predator data points (triangles)
% 
% % Customize the plot
% xlabel('Time');
% ylabel('Population');
% legend('Prey (Deterministic)', 'Predator (Deterministic)', 'Prey Data', 'Predator Data', 'Location', 'Best');
% title('Lotka-Volterra Trajectories and Sampled Data Points');
% grid on;
% hold off;
% 






























% % Define initial conditions and parameters
% x0 = [1.0; 0.5];
% tspan = [0, 15];
% a = 2.0;
% b = 0.8;
% 
% % Solve the differential equation
% [t, sol] = ode45(@(t, y) LV(t, y, a, b), tspan, x0);
% 
% % Generate target data by sampling 15 points and adding Gaussian noise
% sample_times = 1.0:2.0:15.0;
% sampled_sol = interp1(t, sol, sample_times);  % Interpolate solution at sample times
% x = sampled_sol(:, 1) + randn(size(sample_times));  % Add Gaussian noise
% y = sampled_sol(:, 2) + randn(size(sample_times));
% target_data = [x; y];
% 
% % ABC-SMC setup
% nparticles = 1000;
% max_iterations = 1e6;
% convergence = 0.001;
% 
% % Prior distributions for the parameters (uniform between 0 and 5)
% prior_dist = @(n) [unifrnd(0, 5, [n, 1]), unifrnd(0, 5, [n, 1])];
% 
% % Run the ABC-SMC algorithm to estimate parameters
% % Placeholder implementation for an ABC-SMC algorithm
% % This would involve running simulations, calculating distances, and
% % selecting particles with the lowest distances.
% % Note: Actual implementation of ABC-SMC requires a more complex structure.
% 
% % Plotting the results
% figure;
% plot(t, sol(:, 1), 'b', 'DisplayName', 'Prey (Model)');
% hold on;
% plot(t, sol(:, 2), 'r', 'DisplayName', 'Predator (Model)');
% 
% % Plot target data with noise
% plot(sample_times, x, 'bo', 'DisplayName', 'Observed Prey');
% plot(sample_times, y, 'ro', 'DisplayName', 'Observed Predator');
% xlabel('Time');
% ylabel('Population');
% legend;
% title('Lotka-Volterra Model and Target Data');
% 
% % Define Lotka-Volterra ODE function
% function dydt = LV(t, y, a, b)
%     x = y(1);
%     y = y(2);
%     dx = a * x - b * x * y;
%     dy = b * x * y - y;
%     dydt = [dx; dy];
% end

























































































% % Define the true mean and generate observed data
% % Parameters for ABC-SMC
% npart = 50;  % Number of particles
% ngen = 3;  % Number of generations
% tolerances = [50, 25, 10];  % Initial tolerances for each generation
% initial_tolerance = 50;  % Initial tolerance
% 
% % Define priors
% priors = struct('parnames', {'beta', 'gamma'}, 'dist', {'gamma', 'gamma'}, 'p1', [10, 10], 'p2', [10000, 100]);
% 
% % Simulate function, taking parameters, initial state, etc.
% % Placeholder for running the model and returning summary statistics
% 
% % Run ABC-SMC
% % Data and initial states (adjust based on your specific context)
% observed_data = [30, 76];  % Final size and time
% initial_state = [119, 1, 0];  % S, I, R
% 
% % Initialize particles, weights, and outputs for each generation
% particles = cell(ngen, 1);
% outputs = cell(ngen, 1);
% weights = cell(ngen, 1);
% 
% % ABC-SMC loop for each generation
% for gen = 1:ngen
%     tol = tolerances(gen);  % Current generation's tolerance
%     curr_particles = zeros(npart, 2);  % Two parameters (beta, gamma)
%     curr_weights = ones(npart, 1) / npart;  % Initialize weights
%     curr_outputs = zeros(npart, 2);  % Final size and final time
% 
%     for i = 1:npart
%         % Sample from prior distribution (for beta and gamma)
%         beta = gamrnd(priors.p1(1), 1/priors.p2(1));
%         gamma = gamrnd(priors.p1(2), 1/priors.p2(2));
% 
%         % Simulate and check if the output matches the observed data within tolerance
%         result = simSIR([beta, gamma], observed_data, tol, initial_state);
% 
%         % If the result is not NaN, accept it
%         while isnan(result)
%             beta = gamrnd(priors.p1(1), 1/priors.p2(1));
%             gamma = gamrnd(priors.p1(2), 1/priors.p2(2));
%             result = simSIR([beta, gamma], observed_data, tol, initial_state);
%         end
% 
%         curr_particles(i, :) = [beta, gamma];
%         curr_outputs(i, :) = result;
%     end
% 
%     % Store results in the corresponding generation
%     particles{gen} = curr_particles;
%     outputs{gen} = curr_outputs;
%     weights{gen} = curr_weights;  % Uniform weights (modify for reweighting logic)
% end
% 
% % Display results for each generation
% for gen = 1:ngen
%     fprintf('Generation %d:\n', gen);
%     disp('Particles:');
%     disp(particles{gen});
%     disp('Outputs:');
%     disp(outputs{gen});
%     disp('Weights:');
%     disp(weights{gen});
% end
% 
% function stats = simSIR(pars, data, tolerance, iniStates)
%     % Simulate model (replace with your model function)
%     % This is a simple example that must be replaced with actual model implementation
%     % Example: [status, final_time, S, I, R] = run_sir_model(pars, data, iniStates);
% 
%     % Here, the following line simulates a model (hypothetically)
%     status = 1;  % Dummy status
%     final_time = randi([70, 80], 1);  % Random final time
%     final_size = randi([25, 35], 1);  % Random final size
% 
%     % Return vector of summary statistics if the simulation matches the tolerance
%     if all(abs([final_size, final_time] - data) <= tolerance)
%         stats = [final_size, final_time];
%     else
%         stats = NaN;
%     end
% end
