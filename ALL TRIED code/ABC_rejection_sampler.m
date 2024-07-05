clear
close all
clc

% Simulating observed data
% to get the same sequence of random numbers.
rng(123); % Set seed for reproducibility

true_mean = 5;
%  generates 100 random numbers from a normal distribution with a mean 5
%  sigma 1
observed_data = normrnd(true_mean, 1, [1, 100]);

% ABC parameters
N = 1000;  % number of accepted particles
epsilon = 0.1;  % constant tolerance level
prior_min = -10;  % U[-10,10] prior
prior_max = 10;  

% Distance function
calc_distance = @(data1, data2) abs(mean(data1) - mean(data2));

% ABC Rejection Sampling algorithm
res = zeros(N, 2);  % results matrix
i = 1;  % counter for accepted particles
j = 1;  % counter for total proposals

tic;
while i <= N
    % Sample from the prior
    theta_star = unifrnd(prior_min, prior_max);
    
    % Simulate data set from the model
    simulated_data = normrnd(theta_star, 1, [1, 100]);
    
    % Calculate distance
    distance = calc_distance(simulated_data, observed_data);
    
    if distance <= epsilon
        % Store results
        res(i, :) = [theta_star, distance];
        i = i + 1;  % update accepted particle counter
    end
    j = j + 1;  % update total proposal counter
    fprintf('Current acceptance rate = %.6f\r', i / j);  % print acceptance rate
end
toc;
% Plot histogram
subplot(1, 2, 1); % Divide the plot area into 1 row and 2 columns, and select the first subplot
histogram(res(:, 1)); % Plot histogram of the first column (theta_star)
xlabel('\mu', 'Interpreter', 'tex'); % Label x-axis with mu symbol(
ylabel('Frequency')
title('ABC Rejection Sampling'); % Set title for the subplot

% Add a red dotted line at mu = 5
hold on; % Keep the current plot and add new plots
line([5 5], ylim, 'Color', 'r', 'LineStyle', '--'); % Draw a vertical line at x = 5
hold off; % Release the hold on the plot