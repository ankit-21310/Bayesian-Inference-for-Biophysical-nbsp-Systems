
clear;
close all;
clc;

rng(42); % Set the random seed for reproducibility

% Lotka-Volterra ODE system
lv_ode = @(t, y, a, b) [a * y(1) - y(1) * y(2); b * y(1) * y(2) - y(2)];

% ABC-SMC parameters
T = 3; % Number of generations
N = 100; % Number of particles in each generation
epsilon = [30.0, 16.0, 6.0, 5.0, 4.3]; % Tolerance levels for each generation
prior_min = -10; % Prior distribution boundaries
prior_max = 10; % Uniform prior [-10, 10]
perturb_variance = 0.1; % Perturbation variance for random walk

% Observed data from the Lotka-Volterra model
a = 1; % True parameter 'a'
b = 1; % True parameter 'b'
x0 = [1.0; 0.5]; % Initial conditions x(0)=1 and y(0)=0.5
tspan = [0, 15]; % Time span for solving the ODE

% Solve the deterministic Lotka-Volterra system
[t_, Y] = ode15s(@(t_, Y) lv_ode(t_, Y, a, b), tspan, x0);

% Sample data points from the trajectories and add Gaussian noise
sample_times = linspace(0, 15, 8); % Times to sample data
prey_data = interp1(t_, Y(:, 1), sample_times) + randn(1, 8) * 0.5; % Noisy prey data
predator_data = interp1(t_, Y(:, 2), sample_times) + randn(1, 8) * 0.5; % Noisy predator data

% Our observed data x_d, y_d are:
x_d = prey_data;
y_d = predator_data;

% Distance function to calculate squared differences
calc_distance = @(x, y, x_d, y_d) sum((x - x_d).^2) + sum((y - y_d).^2);

% Particles and weights

data_generation_steps = zeros(1,T);
weights = zeros(T, N); % Weights for each population
pop_thetas = zeros(T, N, 2); % Particles (each element is a 2-element vector)

% ABC-SMC algorithm
for t = 1:T

    i = 1;
    j = 0; % data generation step in population t
    while i<= N
        x = zeros(1,8);
        y = zeros(1,8);
        theta_s_s = [0,0];
        
        if t == 1
            % directly sample theta_s_s from prior
            theta_s_s = [unifrnd(prior_min, prior_max), unifrnd(prior_min, prior_max)];
            
        else
            % sample theta_s from previous population
            [max_weight, max_index] = max(weights(t-1, :)); % Returns the max weight and its index

            theta_s = pop_thetas(t-1,max_index, :);  % corresponding theta from previous population

            perturbation = perturb_variance * unifrnd(-1, 1, [1, 2]);

            % theta_s_s kernal Kt(theta/theta_s)
            theta_s_s = theta_s + perturbation ; %perturb the particle   ?????????
 
        end

         % simulate dataset X_star from the model 
         [t_ , Y] = ode15s(@(t_ , Y) lv_ode(t_, Y, theta_s_s(1), theta_s_s(2)), tspan, x0);
         simulated_prey_data = interp1(t_, Y(:, 1), sample_times); % Prey data x
         simulated_predator_data = interp1(t_, Y(:, 2), sample_times); % Predator data y
         x = simulated_prey_data;
         y = simulated_predator_data;
         j = j + 1;

         %fprintf('Current j  = %i\r',  j);

        % calculate distance d(X_star,Xo)
        distance = calc_distance(x, y, x_d, y_d);
        
        


        if distance < epsilon(t)

            % accept the particle theta_s_s store in the res
            pop_thetas(t,i, :) = theta_s_s;
            
            % calculate weight of the particle wi of t
            if t == 1
                weights(t, i) = 1;
            else
                numerator = unifrnd(prior_min, prior_max);
                deominator = 0;
                for k = 1:N
                    deominator = deominator + weights(t-1,k)*unifrnd(-1, 1)*perturb_variance ; 
                end 

                weights(t, i) = numerator/deominator;
            end

            % increment i 
            i = i +1;
        end

    end 

    % store the data generation steps j  
    data_generation_steps(t) = j;
   

    % normalize the weight wt
    row_sum = sum(weights(t, :)); % Calculate the row's total sum
    weights(t, :) = weights(t, :) / row_sum;
    

end


% Create a figure to plot the histograms
figure;

% Plot histograms for `a` and `b` in each subplot
for t = 1:T
    subplot(T, 1, t); % Create a subplot for each population (T rows, 1 column)
    
    % Extract the `a` and `b` values for the current population
    a_values = pop_thetas(t, :, 1); % All `a` values for population `t`
    b_values = pop_thetas(t, :, 2); % All `b` values for population `t`
    
    % Plot histograms of `a` and `b` in the same subplot
    hold on;
    histogram(a_values, 'DisplayName', 'Parameter a', 'FaceColor', 'b', 'FaceAlpha', 0.5); % Histogram for `a`
    histogram(b_values, 'DisplayName', 'Parameter b', 'FaceColor', 'r', 'FaceAlpha', 0.5); % Histogram for `b`
    hold off;
    
    % Add labels, title, and legend
    xlabel('Parameter Value');
    ylabel('Frequency');
    title(['Population ', num_str(t)]);
    legend show; % Show legend
end