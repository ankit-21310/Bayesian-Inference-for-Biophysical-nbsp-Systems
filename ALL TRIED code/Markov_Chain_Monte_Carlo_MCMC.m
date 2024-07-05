clear
close all
clc
% 
% % Parameters
% num_samples = 100;  % Number of samples to generate
% initial_state = 0;    % Initial state of the Markov chain
% proposal_sd = 1;      % Standard deviation of the proposal distribution (Gaussian)
% 
% % Function to compute the probability density of a state in the target distribution
% target_distribution = @(x) exp(-0.5 * (x - 3)^2) / sqrt(2 * pi); % Example: N(3,1)
% 
% 
% samples = zeros(1, num_samples);
% current_state = initial_state;
% 
% % Generate samples using Metropolis-Hastings algorithm
% for i = 1:num_samples
%     % Propose a new state from a Gaussian distribution
%     proposed_state = current_state + proposal_sd * randn;
% 
%     % Compute the acceptance probability
%     acceptance_prob = min(1, target_distribution(proposed_state) / target_distribution(current_state));
% 
%     % Accept or reject the proposed state
%     if rand < acceptance_prob
%         current_state = proposed_state;
%     end
% 
%     % Save the current state as a sample
%     samples(i) = current_state;
% 
%     % Plot sample vs iteration
%     if mod(i, 100) == 0 % Plot every 100 iterations
%         subplot(2,1,1);
%         histogram(samples(1:i), 'Normalization', 'probability', 'BinWidth', 0.2);
%         xlabel('Sample Value');
%         ylabel('Probability');
%         title('Histogram of Samples from Gaussian Distribution');
%         subplot(2,1,2);
%         plot(1:i, samples(1:i));
%         xlabel('Iteration');
%         ylabel('Sample Value');
%         title('Sample Value vs Iteration');
%         drawnow;
%     end
% end
% 
% % Plot the final histogram of generated samples
% subplot(2,1,1);
% histogram(samples, 'Normalization', 'probability', 'BinWidth', 0.2);
% xlabel('Sample Value');
% ylabel('Probability');
% title('Histogram of Samples from Gaussian Distribution');
% hold on;
% subplot(2,1,2);
% plot(1:num_samples, samples);
% xlabel('Iteration');
% ylabel('Sample Value');
% title('Sample Value vs Iteration');

% % % % % % % % clear
% % % % % % % % close all
% % % % % % % % clc
% % % % % % % % 
% % % % % % % % % Parameters
% % % % % % % % num_samples = 100;  % Number of samples to generate
% % % % % % % % initial_state = 0;    % Initial state of the Markov chain
% % % % % % % % proposal_sd = 1;      % Standard deviation of the proposal distribution (Gaussian)
% % % % % % % % 
% % % % % % % % % Function to compute the probability density of a state in the target distribution
% % % % % % % % target_distribution = @(x) exp(-0.5 * ((x - 3)^2)) / sqrt(2 * pi); % Example: N(3,1)
% % % % % % % % 
% % % % % % % % samples = zeros(1, num_samples);
% % % % % % % % current_state = initial_state;
% % % % % % % % 
% % % % % % % % % Generate samples using Metropolis-Hastings algorithm
% % % % % % % % for i = 1:num_samples
% % % % % % % %     % Propose a new state from a Gaussian distribution
% % % % % % % %     proposed_state = current_state + proposal_sd * randn;
% % % % % % % % 
% % % % % % % %     % Compute the acceptance probability
% % % % % % % %     acceptance_prob = min(1, target_distribution(proposed_state) / target_distribution(current_state));
% % % % % % % % 
% % % % % % % %     % Accept or reject the proposed state
% % % % % % % %     if rand < acceptance_prob
% % % % % % % %         current_state = proposed_state;
% % % % % % % %     end
% % % % % % % % 
% % % % % % % %     % Save the current state as a sample
% % % % % % % %     samples(i) = current_state;
% % % % % % % % 
% % % % % % % %     % Plot sample vs iteration
% % % % % % % %     if mod(i, 100) == 0 % Plot every 100 iterations
% % % % % % % %         subplot(2,1,1);
% % % % % % % %         histogram(samples(1:i), 'Normalization', 'probability', 'BinWidth', 0.2);
% % % % % % % %         xlabel('Sample Value');
% % % % % % % %         ylabel('Probability');
% % % % % % % %         title('Histogram of Samples from Gaussian Distribution');
% % % % % % % %         subplot(2,1,2);
% % % % % % % %         plot(1:i, samples(1:i));
% % % % % % % %         xlabel('Iteration');
% % % % % % % %         ylabel('Sample Value');
% % % % % % % %         title('Sample Value vs Iteration');
% % % % % % % %         drawnow;
% % % % % % % %     end
% % % % % % % % end
% % % % % % % % 
% % % % % % % % % Plot the final histogram of generated samples
% % % % % % % % subplot(2,1,1);
% % % % % % % % histogram(samples, 'Normalization', 'probability', 'BinWidth', 0.2);
% % % % % % % % xlabel('Sample Value');
% % % % % % % % ylabel('Probability');
% % % % % % % % title('Histogram of Samples from Gaussian Distribution');
% % % % % % % % hold on;
% % % % % % % % subplot(2,1,2);
% % % % % % % % plot(1:num_samples, samples);
% % % % % % % % xlabel('Iteration');
% % % % % % % % ylabel('Sample Value');
% % % % % % % % title('Sample Value vs Iteration');
clear
close all
clc

% Parameters
num_samples = 10000;  % Number of samples to generate
initial_state = 0;    % Initial state of the Markov chain
proposal_sd = 1;      % Standard deviation of the proposal distribution (Gaussian)

% Function to compute the probability density of a state in the target distribution
target_distribution = @(x) exp(-0.5 * ((x - 3).^2)) / sqrt(2 * pi); % Example: N(3,1)

samples = zeros(1, num_samples);
current_state = initial_state;

% Generate samples using Metropolis-Hastings algorithm
for i = 1:num_samples
    % Propose a new state from a Gaussian distribution
    proposed_state = current_state + proposal_sd * randn;

    % Compute the acceptance probability
    acceptance_prob = min(1, target_distribution(proposed_state) / target_distribution(current_state));

    % Accept or reject the proposed state
    if rand < acceptance_prob
        current_state = proposed_state;
    end

    % Save the current state as a sample
    samples(i) = current_state;
end

% Plot the target distribution
x = linspace(min(samples), max(samples), 1000);
y_target = target_distribution(x);
figure;
plot(x, y_target, 'b-', 'LineWidth', 2);
hold on;

% Plot the histogram of generated samples
histogram(samples, 'Normalization', 'probability', 'BinWidth', 0.2, 'FaceColor', [0.85, 0.33, 0.1]);

xlabel('Sample Value');
ylabel('Probability');
title('Target Distribution and Sampled Distribution');
legend('Target Distribution', 'Sampled Distribution');
