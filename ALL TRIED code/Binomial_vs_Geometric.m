clear
close all
clc

% % 
% % 
% % % Define the parameters for Binomial Distribution
% % N = [10, 20, 50, 100]; % number of trials
% % p = 0.5; % probability of success
% % 
% % % Define the parameters for Geometric Distribution
% % p_geom = 0.2; % probability of success
% % 
% % % Create a figure with 2x2 subplots
% % figure;
% % 
% % % Loop through the different values of N
% % for i = 1:length(N)
% %     % Generate Binomial and Geometric random numbers
% %     x_binom = binornd(N(i), p, 1, 10000); % generate 10000 random numbers from Binomial Distribution
% %     x_geom = geornd(p_geom, 1, 10000); % generate 10000 random numbers from Geometric Distribution
% % 
% %     % Compute the empirical distribution of Binomial and Geometric random numbers
% %     freq_binom = histc(x_binom, 0:N(i)); % compute the frequency of each outcome
% %     freq_geom = histc(x_geom, 0:100); % compute the frequency of each outcome
% % 
% %     % Normalize the frequency to obtain the probability mass function
% %     pmf_binom = freq_binom / sum(freq_binom); % normalize the frequency
% %     pmf_geom = freq_geom / sum(freq_geom); % normalize the frequency
% % 
% %     % Plot the probability mass function of Binomial and Geometric Distributions
% %     subplot(2, 2, i);
% %     bar(0:N(i), pmf_binom, 'FaceColor', [0.8 0.8 0.8]); % plot the pmf of Binomial Distribution
% %     hold on;
% %     bar(0:100, pmf_geom, 'FaceColor', [0.8 0.2 0.2]); % plot the pmf of Geometric Distribution
% %     xlabel('Number of Successes');
% %     ylabel('Probability');
% %     title(['Binomial Distribution with N = ', num2str(N(i))]);
% %     hold off;
% % end
% 
% 
% % Define the parameters for Binomial Distribution
% N = [10, 20, 50, 100]; % number of trials
% p = 0.5; % probability of success
% 
% % Define the parameters for Geometric Distribution
% p_geom = 0.2; % probability of success
% 
% % Create a figure with 2x2 subplots
% figure;
% 
% % Loop through the different values of N
% for i = 1:length(N)
%     % Generate Binomial and Geometric random numbers
%     x_binom = binornd(N(i), p, 1, 10000); % generate 10000 random numbers from Binomial Distribution
%     x_geom = geornd(p_geom, 1, 10000); % generate 10000 random numbers from Geometric Distribution
% 
%     % Compute the empirical distribution of Binomial and Geometric random numbers
%     freq_binom = histc(x_binom, 0:N(i)); % compute the frequency of each outcome
%     freq_geom = histc(x_geom, 0:100); % compute the frequency of each outcome
% 
%     % Normalize the frequency to obtain the probability mass function
%     pmf_binom = freq_binom / sum(freq_binom); % normalize the frequency
%     pmf_geom = freq_geom / sum(freq_geom); % normalize the frequency
% 
%     % Plot the probability mass function of Binomial and Geometric Distributions
%     subplot(2, 2, i);
%     bar(0:N(i), pmf_binom, 'FaceColor', [0.8 0.8 0.8]); % plot the pmf of Binomial Distribution
%     hold on;
%     bar(0:100, pmf_geom, 'FaceColor', [0.8 0.2 0.2]); % plot the pmf of Geometric Distribution
%     xlabel('Number of Trials/Successes');
%     ylabel('Probability');
%     title(['Comparison of Binomial and Geometric Distributions with N = ', num2str(N(i))]);
%     legend('Binomial', 'Geometric', 'Location', 'best');
%     hold off;
% end


% Define the parameters for Geometric Distribution
p_geom = 0.2; % probability of success

% Define mean lambda for Poisson Distribution
lambda = 10; % You can change this value to observe different convergences

% Calculate the corresponding p for given lambda
p = lambda / 10; % Since N = 10 for all cases

% Define parameters for the Binomial Distribution
N_values = [10, 50, 100, 500]; % number of trials

% Create a figure with subplots
figure;

% Plot Geometric Distribution PMF
k = 1:20; % Range of values for Geometric Distribution
pmf_geom = (1 - p_geom).^(k - 1) * p_geom; % PMF of Geometric Distribution
subplot(1, 2, 1);
bar(k, pmf_geom, 'FaceColor', [0.8 0.2 0.2]); % Plot PMF
xlabel('Number of Trials until First Success');
ylabel('Probability');
title('Geometric Distribution');

% Plot Binomial Distribution PMF
subplot(1, 2, 2);
for i = 1:length(N_values)
    N = N_values(i);
    p = lambda / N; % Adjust p to keep lambda constant
    
    % Compute PMF of Binomial Distribution
    pmf_binom = binopdf(k, N, p);
    
    % Plot PMF
    plot(k, pmf_binom, 'o-', 'DisplayName', ['N = ' num2str(N)]);
    hold on;
end
xlabel('Number of Successes');
ylabel('Probability');
title('Binomial Distribution Converging to Poisson');
legend('Location', 'best');
hold off;

