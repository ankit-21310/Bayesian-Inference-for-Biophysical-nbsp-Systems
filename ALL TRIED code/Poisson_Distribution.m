clear
close all
clc
 
% 
p = 0.5;
n = 200;
lambda = n*p;
k_values = 1:n; % Range of values for k

% %claulating factorial of n using stirling approximation
fact = @(n) (sqrt(2*pi*n)) * ( (n/exp(1) )^n);

% % poisson pfm
poisson_pmf = @(k) ((exp(-lambda) * lambda^k)) / fact(k);
           

% % Calculating PMF values
Poisson_pmf_values = zeros(1,n);
for k = 1:n
    Poisson_pmf_values(k) = poisson_pmf(k);
end

% % Plotting
% stem(k_values, Poisson_pmf_values, 'LineStyle', '-', 'MarkerFaceColor', 'blue', 'LineWidth', 1.3);
% % Plotting the Poisson distribution
% xlabel('Number of Successes (k)');
% ylabel('Probability: P(k)');
% title( 'Poisson Distribution ');
% legend(sprintf('n = %d  p = %.2f lambda = %.2f', n,p,lambda));


% computing binomial coefficient with the help of gamma function
binomial_coefficient = @(n, k) exp(gammaln(n+1) - gammaln(k+1) - gammaln(n-k+1));

% Define the anonymous function for the PMF of the binomial distribution
binomial_pmf = @(n, k, p) binomial_coefficient(n, k) *pow(p,k) * pow((1 - p),(n - k));

no_success = 1:n;
pmf_vector = zeros(1, n); % to store the pdf for each k;

for k = 1:n
    pmf_vector(k) = binomial_pmf(n,k,p);
end

subplot(1, 2, 1);
% Plotting the Poisson distribution
stem(k_values, Poisson_pmf_values, 'LineStyle', '-', 'MarkerFaceColor', 'blue', 'LineWidth', 1);

xlabel('Number of Successes (k)');
ylabel('Probability: P(k)');
title( 'Poisson Distribution ');
legend(sprintf('n = %d  p = %.2f lambda = %.2f', n,p,lambda));
grid on;

subplot(1, 2, 2);
stem(k_values, pmf_vector, 'LineStyle', '-', 'MarkerFaceColor', 'red', 'LineWidth', 1);
xlabel('Number of Successes');
ylabel('Probability');
title('Binomial Distribution');
legend(sprintf('n = %d  p1 = %.2f', n,p))
grid on;






%Definition of the pow function
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

