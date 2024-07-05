clear;
close all;
clc;

% Parameters for the distributions
p = 0.58; % Probability of success

% Range for n values
n_values = [5, 50, 50, 100];

figure;
for i = 1:length(n_values)
    % Calculate binomial samples
    n = n_values(i);
    samples = binornd(n, p, 1000, 1); % Generate 1000 samples
    
    % Calculate Gaussian distribution
    mu = n * p;
    sigma = sqrt(n * p * (1 - p));
    x = -10:0.1:70; % Redefine x range
    gaussian_distribution = 1/(sigma * sqrt(2*pi)) * exp(-(x-mu).^2 / (2*sigma^2));

    % Plot histogram of binomial samples
    subplot(2, 2, i);
    histogram(samples, 'Normalization', 'pdf', 'DisplayName', 'Binomial');
    hold on;
    plot(x, gaussian_distribution, 'r', 'LineWidth', 2, 'DisplayName', 'Gaussian');
    hold off;

    % Add labels and legend
    title(['n = ', num2str(n)]);
    xlabel('x');
    ylabel('Probability');
    legend('Location', 'best');
    grid on;
end

% Adjust the layout
sgtitle('Binomial vs Gaussian Distribution with Varying n (Histogram)');

% clear
% close all
% clc
% 
% 
% 
% 
% % Parameters for the distributions
% p = 0.5; % Probability of success
% 
% % Range for n values
% n_values = [5, 50, 50, 100];
% 
% % Range for x values
% x = -10:0.1:60;
% 
% figure;
% for i = 1:length(n_values)
%     % Calculate binomial distribution
%     n = n_values(i);
%     binomial_distribution = binopdf(x, n, p);
% 
%     % Calculate Gaussian distribution
%     mu = n * p;
%     sigma = sqrt(n * p * (1 - p));
%     gaussian_distribution = 1/(sigma * sqrt(2*pi)) * exp(-(x-mu).^2 / (2*sigma^2));
% 
%     % Plot both distributions
%     subplot(2, 2, i);
%     plot(x, binomial_distribution, 'b', 'DisplayName', 'Binomial');
%     hold on;
%     plot(x, gaussian_distribution, 'r', 'LineWidth', 2, 'DisplayName', 'Gaussian');
%     hold off;
% 
%     % Add labels and legend
%     title(['n = ', num2str(n)]);
%     xlabel('x');
%     ylabel('Probability');
%     legend('Location', 'best');
%     grid on;
% end
% 
% % Adjust the layout
% sgtitle('Binomial vs Gaussian Distribution with Varying n');
