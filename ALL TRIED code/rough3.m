
clear
close all
clc



% First, create some fake data from a mixture of two exponential distributions
fakeData = [exprnd(1, 50, 1); exprnd(0.01, 150, 1)];

% Now let's run our Gibbs sampler for our Exponential Mixture Model
Result = ExpMix_Gibbs(fakeData, 500, [1, 1], 2);



% 
% 
% 
% % Create some fake data
% fakeData = [exprnd(1, 50, 1); exprnd(0.01, 150, 1)];
% 
% % Run Gibbs sampler
% Result = ExpMix_Gibbs(fakeData, 500, [1, 1], 2);
% 
% % % % % Plot results
% % % figure;
% % % mixturePlot(fakeData, Result, 20);
% % % xlabel('Data (Log Scale)');
% % % ylabel('Counts');
% % % set(gca, 'FontSize', 12);


% Define layout for subplots
figure;

% First row
subplot(4, 2, 1);
plot(Result.theta(:, 1), 'Color', [0, 0, 0.85], 'LineWidth', 1.5);
ylabel('\theta_1');
xlabel('MCMC Iteration');
title('\theta_1 Evolution');
set(gca, 'FontSize', 12);
set(gca, 'LineWidth', 1.5);

subplot(4, 2, 2);
plot(Result.theta(:, 2), 'Color', [0, 0, 0.85], 'LineWidth', 1.5);
ylabel('\theta_2');
xlabel('MCMC Iteration');
title('\theta_2 Evolution');
set(gca, 'FontSize', 12);
set(gca, 'LineWidth', 1.5);

% Second row
subplot(4, 2, 3);
histogram(Result.theta(50:end, 1), 'Normalization', 'probability', 'FaceColor', [0.21, 0.56, 0.75]);
xlabel('\theta_1');
ylabel('Probability Density');
title('\theta_1 Posterior');
set(gca, 'FontSize', 12);
set(gca, 'LineWidth', 1.5);
hold on;
xline(0.01, 'LineWidth', 4, 'Color', [0.21, 0.56, 0.75]);

subplot(4, 2, 4);
histogram(1./Result.theta(50:end, 2), 'Normalization', 'probability', 'FaceColor', [0.21, 0.56, 0.75]);
xlabel('\theta_2');
ylabel('Probability Density');
title('\theta_2 Posterior');
set(gca, 'FontSize', 12);
set(gca, 'LineWidth', 1.5);
hold on;
xline(1, 'LineWidth', 4, 'Color', [0.21, 0.56, 0.75]);

% Third row
subplot(4, 2, 5);
plot(Result.weights(:, 1), 'Color', [0, 0, 0.85], 'LineWidth', 1.5);
ylabel('w_1');
xlabel('MCMC Iteration');
title('w_1 Evolution');
set(gca, 'FontSize', 12);
set(gca, 'LineWidth', 1.5);

subplot(4, 2, 6);
plot(Result.weights(:, 2), 'Color', [0, 0, 0.85], 'LineWidth', 1.5);
ylabel('w_2');
xlabel('MCMC Iteration');
title('w_2 Evolution');
set(gca, 'FontSize', 12);
set(gca, 'LineWidth', 1.5);

% Fourth row
subplot(4, 2, 7);
histogram(Result.weights(50:end, 1), 'Normalization', 'probability', 'FaceColor', [0.21, 0.56, 0.75]);
xlabel('w_1');
ylabel('Probability Density');
title('w_1 Posterior');
set(gca, 'FontSize', 12);
set(gca, 'LineWidth', 1.5);
hold on;
xline(0.75, 'LineWidth', 4, 'Color', [0.21, 0.56, 0.75]);

subplot(4, 2, 8);
histogram(Result.weights(50:end, 2), 'Normalization', 'probability', 'FaceColor', [0.21, 0.56, 0.75]);
xlabel('w_2');
ylabel('Probability Density');
title('w_2 Posterior');
set(gca, 'FontSize', 12);
set(gca, 'LineWidth', 1.5);
hold on;
xline(0.25, 'LineWidth', 4, 'Color', [0.21, 0.56, 0.75]);








% Function for drawing Dirichlet random variables
function x = rdirichlet(n, alpha)
    l = length(alpha);
    x = zeros(n, l);
    for i = 1:n
        x(i, :) = gamrnd(alpha, 1);
    end
    sm = sum(x, 2);
    x = x ./ sm;
end

% Function for drawing from the conditional posterior of the scale parameter Theta
function theta = cp_theta(dat, priors)
    if isempty(dat)
        A = priors(1);
        B = priors(2);
    else
        A = priors(1) + length(dat);
        B = priors(2) + sum(dat);
    end
    theta = gamrnd(A, 1 / B);
end

% Function for drawing from the conditional posterior of the mixture weights
function Pi = cp_Pi(dat, z, k)
    N_in_each = zeros(1, k);
    for l = 1:k
        N_in_each(l) = length(dat(z == l));
    end
    Pi = rdirichlet(1, N_in_each);
end

% Function for drawing from the conditional posterior of latent variables z
function z = cp_z(dat, Pi, theta)
    n = length(dat);
    z = zeros(1, n);
    for t = 1:n
        probs = zeros(1, length(theta));
        for l = 1:length(theta)
            probs(l) = Pi(l) * exppdf(dat(t), 1 / theta(l));
        end
        probs = probs / sum(probs);
        [~, z(t)] = max(mnrnd(1, probs));
    end
end



% Function for iteratively sampling from the relevant conditional posteriors
function Results = ExpMix_Gibbs(dat, Iterations, pr, k)
    % Arguments:
    %   - data vector
    %   - desired number of iterations
    %   - hyperparameters for priors
    %   - number of components
    % Values: A struct of Results - includes all posterior samples for Thetas, Pi, and z
    
    % Hyperparameters
    priors = pr;
    
    % Initialize parameters
    theta = NaN(Iterations, k);
    theta(1, :) = gamrnd(priors(1), 1 / priors(2), 1, k);
    
    Pi = NaN(Iterations, k);
    Pi(1, :) = rdirichlet(1, ones(1, k));
    
    [n, ~] = size(dat);
    z = NaN(Iterations, n);
    
    % Main loop
    for i = 2:Iterations
        z(i, :) = cp_z(dat, Pi(i - 1, :), theta(i - 1, :));
        for j = 1:k
            Relevant_Data = dat(z(i, :) == j);
            theta(i, j) = cp_theta(Relevant_Data, priors);
        end
        Pi(i, :) = cp_Pi(dat, z(i, :), k);
    end
    
    Results = struct('theta', theta, 'weights', Pi, 'occ', z);
end


