
clear
close all
clc 

n = 200;  % no of trials 
p = 0.2;  % probalility of success

% computing binomial coefficient with the help of gamma function
binomial_coefficient = @(n, k) exp(gammaln(n+1) - gammaln(k+1) - gammaln(n-k+1));

% Define the anonymous function for the PMF of the binomial distribution
binomial_pmf = @(n, k, p) binomial_coefficient(n, k) *pow(p,k) * pow((1 - p),(n - k));

no_success = 1:n;
pmf_vector = zeros(1, n); % to store the pdf for each k;

for k = 1:n
    pmf_vector(k) = binomial_pmf(n,k,p);
end

stem(no_success, pmf_vector, 'LineStyle', '-', 'MarkerFaceColor', 'red', 'LineWidth', 1.5);

%Plot the binomial distribution (PMF)
xlabel('Number of Successes');
ylabel('Probability');
title('Binomial Distribution PMF');
grid on;




% This fun is binary exponentiation to compute a^n in log(n) timeComplexity
function result = pow(a, x)
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
