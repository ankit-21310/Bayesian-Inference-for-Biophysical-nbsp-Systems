clear
close all
clc

%MCMC WITH CAUCHY DISTRIBUTION

% Parameters
x0 = 0;         % Initial value
num_samples = 1000; % Number of samples to generate
burn_in = 100;  % Number of burn-in iterations

% Initialization
x = zeros(num_samples, 1); % Samples
x(1) = x0; % Starting sample
acceptance_rate = 0; % Initialize acceptance rate

% Metropolis-Hastings Algorithm
for t = 2:num_samples
    % Proposal: generate a sample from a Cauchy distribution
    % centered at the current sample (x(t-1))
    x_proposed = x(t-1) + 0.1 * randn() * cauchy_rand(); % Using a normal distribution as proposal
    
    % Calculate the acceptance probability
    alpha = min(1, cauchy_pdf(x_proposed) / cauchy_pdf(x(t-1)));
    
    % Accept or reject the proposal
    if rand() < alpha
        x(t) = x_proposed; % Accept the proposal
        acceptance_rate = acceptance_rate + 1;
    else
        x(t) = x(t-1); % Reject the proposal
    end
end

% Calculate acceptance rate
acceptance_rate = acceptance_rate / num_samples;

% Plotting
figure;

% Plot the samples
subplot(2,1,1);
plot(x);
title('Samples from the Cauchy Distribution');
xlabel('Iteration');
ylabel('Sample Value');

% Plot histogram of samples
subplot(2,1,2);
histogram(x(burn_in:end), 50, 'Normalization', 'probability');
hold on;
x_values = linspace(min(x(burn_in:end)), max(x(burn_in:end)), 100);
plot(x_values, cauchy_pdf(x_values), 'r', 'LineWidth', 2);
hold off;
title('Histogram of Samples vs. True Cauchy Distribution');
xlabel('Sample Value');
ylabel('Probability Density');

% Display acceptance rate
fprintf('Acceptance rate: %.2f%%\n', acceptance_rate * 100);

% Cauchy Distribution PDF
function p = cauchy_pdf(x)
    p = 1/pi * (1./(1 + x.^2));
end

% Cauchy Distribution Random Number Generator
function r = cauchy_rand()
    r = tan(pi*(rand() - 0.5));
end
