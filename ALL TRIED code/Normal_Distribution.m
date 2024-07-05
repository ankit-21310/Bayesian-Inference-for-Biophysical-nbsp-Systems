
clear
close all
clc


% pdf of Gaussian distribution
gaussian = @(x, mean, variance) (1 / sqrt(2 * pi * variance)) * exp(-(x - mean).^2 / (2 * variance));

% range for x values
x = linspace(-10, 10, 1000);

% different mean and variance values
mean_values = [0, 2, 4];
variance_values = [4, 3.8, 4];

% Plot the Gaussian distributions
figure;
hold on;
for i = 1:length(mean_values)
    mean_val = mean_values(i);
    variance_val = variance_values(i);
    y = gaussian(x,mean_val,variance_val);
    plot(x, y,'LineWidth', 2, 'DisplayName', sprintf('Mean=%d, Variance=%d', mean_val, variance_val));
end

title('Gaussian Normal Distribution');
xlabel('x');
ylabel('Probability Density');
legend('Location', 'best');
grid on;
hold off;
