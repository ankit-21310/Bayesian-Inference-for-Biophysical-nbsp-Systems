clear
close all
clc


% Parameters
lambda = 20; % Poisson parameter (constant)
n_values = [50, 100, 200]; % Different numbers of trials
p_values = lambda ./ n_values; % Probability of success for each n

% Generate x-axis values
x_binomial = 0:40; % Possible outcomes for binomial distribution
x_poisson = 0:max(x_binomial); % Possible outcomes for Poisson distribution

% Custom function for binomial coefficient
binomial_coefficient = @(n, k) exp(gammaln(n+1) - gammaln(k+1) - gammaln(n-k+1));

% Custom function for binomial PMF
binomial_pmf = @(n, k, p) binomial_coefficient(n, k) * pow(p, k) * pow((1 - p), (n - k));

% Custom function for Poisson PMF
poisson_pmf = @(x, lambda) exp(-lambda) .* lambda.^x ./ factorial(x);

% Plot
figure;
for i = 1:length(n_values)
    n = n_values(i);
    p = p_values(i);
    
    % Compute binomial PMF
    pmf_binomial = arrayfun(@(k) binomial_pmf(n, k, p), x_binomial);
    
    % Compute Poisson PMF
    pmf_poisson = poisson_pmf(x_poisson, lambda);
    
    % Plot binomial distribution
    subplot(length(n_values), 1, i);
    stem(x_binomial, pmf_binomial, 'filled', 'LineWidth', 1.5, 'Color', 'b');
    hold on;
    
    % Plot Poisson distribution
    stem(x_poisson, pmf_poisson, 'LineWidth', 1.5, 'Color', 'r');
    
    % Set plot properties
    xlabel('Outcome');
    ylabel('Probability');
    title(['Binomial (n = ', num2str(n), ', p = ', num2str(p), ') vs Poisson (\lambda = ', num2str(lambda), ') Distribution']);
    legend('Binomial', 'Poisson');
    grid on;
end

% Binary exponentiation function to compute a^x in log(x) time complexity
function result = pow(a, x)
if x == 0
    result = 1; % for a^0
    return;
end

result = 1;
while x > 0
    if bitand(x, 1) == 1 % update the result = result * a when LSB = 1
        result = result * a;
    end
    x = bitshift(x, -1); % right shift the power
    a = a * a; % square the base after each right shift
end
end
