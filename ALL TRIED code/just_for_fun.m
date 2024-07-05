clear
close all
clc

% Define the target distribution function
target_distribution = @(x) exp(-0.5 * (x - 3).^2) / sqrt(2 * pi);

% Generate a range of x values
x_values = linspace(-10, 10, 1000);

% Evaluate the target distribution at these x values
y_values = target_distribution(x_values);

% Plot the target distribution curve
plot(x_values, y_values, 'r', 'LineWidth', 2);
xlabel('x');
ylabel('Probability Density');
title('Target Distribution Plot');
