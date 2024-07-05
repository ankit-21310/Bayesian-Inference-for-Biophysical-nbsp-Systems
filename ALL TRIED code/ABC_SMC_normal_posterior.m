clear
close all
clc

% Simulating observed data
rng(123); % Set seed for reproducibility
true_mean = 5;
observed_data = normrnd(true_mean, 1, [1, 100]);

% ABC-SMC parameters
N = 200;  % number of particles
tolerance = linspace(3, 0.1, 5);  % A sequence of decreasing tolerance levels. 
prior_min = -10;  % U[-10,10] prior
prior_max = 10;   

% Distance function
calc_distance = @(data1, data2) abs(mean(data1) - mean(data2));

% ABC-SMC Algorithm
res = zeros(0, 3);  % Include weight in the results matrix
for epsilon = tolerance
    accepted_theta = zeros(1, 0);
    accepted_distances = zeros(1, 0);
    weights = zeros(1, 0);
    
    while length(accepted_theta) < N
        % Sampling from the prior
        theta = unifrnd(prior_min, prior_max);
        
        % Simulating data from the model
        simulated_data = normrnd(theta, 1, size(observed_data));
        
        % Calculating distance
        distance = calc_distance(simulated_data, observed_data);
        
        % Acceptance condition
        if distance < epsilon
            accepted_theta = [accepted_theta, theta];  % store list of accepted theta under current tolerance
            accepted_distances = [accepted_distances, distance]; % store list of accepted distance under current tolerance
            weights = [weights, 1 / (distance + eps)];  % inverse of distance as weight
        end
    end
    
    % Normalize weights
    weights = weights / sum(weights);
    
    % Resampling step: Select indices from accepted particles proportional to weight
    resample_indices = randsample(1:N, N, true, weights); 
    accepted_theta = accepted_theta(resample_indices);
    accepted_distances = accepted_distances(resample_indices);
    weights = repmat(1/N, 1, N);  % Reset uniform weights after resampling
    
    new_entries = [accepted_theta', accepted_distances', weights'];
    res = [res; new_entries];
    disp(['Tolerance: ', num2str(epsilon), ', Mean estimate: ', num2str(mean(accepted_theta))]);
end

% Save results to CSV file for ABC-SMC
writematrix(res, 'ABC_SMC_Results.csv');

% Posterior estimate
posterior_estimate = mean(res(:, 1));  
disp(['Final posterior mean estimate: ', num2str(posterior_estimate)]);




% Read CSV file
abc_smc = csvread('ABC_SMC_Results.csv');

% Plot histogram
histogram(abc_smc(:, 1)); % Plot histogram of the first column (theta)
xlabel('\mu', 'Interpreter', 'tex'); % Label x-axis with mu symbol
ylabel('Frequency'); % Label y-axis as Frequency
title('ABC SMC Sampling'); % Set title for the plot

% Add a red dotted line at mu = 5
hold on; % Keep the current plot and add new plots
line([5 5], ylim, 'Color', 'r', 'LineStyle', '--'); % Draw a vertical line at x = 5
hold off; % Release the hold on the plot
