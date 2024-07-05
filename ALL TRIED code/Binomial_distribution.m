clear
close all
clc

% no of trials 
n1 = 200;
n2 = 100;
n3 = 100;

% probability of success
p1 = 0.6;
p2 = 0.2;
p3 = 0.8;
% Define a range of x values (number of successes)
x_values = 0:max([n1, n2, n3]);

binomial_pmf = @(n, k, p) nchoosek(n, k) * pow(p,k) * pow((1 - p),(n - k));

% Calculate the probability mass function (PMF) values for each value of n
pmf_values1 = binopdf(x_values, n1, p1);
pmf_values2 = binopdf(x_values, n2, p2);
pmf_values3 = binopdf(x_values, n3, p3);


% Plot the binomial distributions (PMFs)
stem(x_values, pmf_values1, 'LineStyle', '-', 'MarkerFaceColor', 'blue', 'LineWidth', 1);
hold on;
stem(x_values, pmf_values2, 'LineStyle', '-', 'MarkerFaceColor', 'red', 'LineWidth', 1);
stem(x_values, pmf_values3, 'LineStyle', '-', 'MarkerFaceColor', 'auto', 'LineWidth', 1);
hold off;

xlabel('Number of Successes');
ylabel('Probability');
title('Binomial Distribution PMF');
legend(sprintf('n = %d  p1 = %.2f', n1,p1), sprintf('n = %d  p2 = %.2f', n2,p2), sprintf('n = %d  p3 = %.2f', n3,p3));
%legend('n = 100  p1 = 0.8', 'n = 30  p2 = 0.3', 'n = 50  p3 = 0.6');
grid on;



% Function definitions
function result = pow(a, x)
% Definition of the pow function
if x == 0
    result = 1; % for a^0
    return;
end

result = 1;
while x > 0
    if bitand(x, 1) == 1 % update the ans = ans*a when LSB = 1
        result = result * a;
    end
    x = bitshift(x, -1); % right shift the power
    a = a * a; % square the base after each right shift
end
end



