

clear
close all
clc
% 
% 
% % First, create some fake data from a mixture of two exponential distributions
% fakeData = [exprnd(1, 1, 50), exprnd(100, 1, 150)];
% 
% % Now let's run our Gibbs sampler for our Exponential Mixture Model
% Result = ExpMix_Gibbs(fakeData, 500, [1, 1], 2);
% 
% Result
% 
% 
% % Set up the layout
% subplot(4, 2, 1);
% 
% % Set up the plot parameters
% end_value = size(Result.theta, 1);
% plot(1:end_value, Result.theta(:, 1), 'LineWidth', 1.5, 'Color', [0 0 0.85]);
% 
% % Set labels and title
% ylabel('\theta_1');
% xlabel('MCMC Iteration');
% set(gca, 'FontSize', 12);
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% function x = rdirichlet(n, alpha)
%     l = length(alpha);
%     x = gamrnd(alpha, 1, n, l);
%     sm = sum(x, 2);
%     x = x ./ sm;
% end
% 
% function theta = cp_theta(dat, priors)
%     if isempty(dat)
%         A = priors(1);
%         B = priors(2);
%     else
%         A = priors(1) + length(dat);
%         B = priors(2) + sum(dat);
%     end
%     theta = gamrnd(A, 1 / B);
% end
% 
% function Pi = cp_Pi(dat, z, k)
%     N_in_each = arrayfun(@(l) sum(dat(z == l)), 1:k);
%     Pi = rdirichlet(1, ones(1, k));
% end
% 
% function z = cp_z(dat, Pi, theta)
%     z = zeros(1, length(dat));
%     for t = 1:length(dat)
%         probs = Pi .* exppdf(dat(t), 1 ./ theta);
%         probs = probs / sum(probs);
%         [~, ind] = max(mnrnd(1, probs));
%         z(t) = ind;
%     end
% end
% 
% function result = ExpMix_Gibbs(dat, Iterations, pr, k)
%     priors = pr;
%     theta = zeros(Iterations, k);
%     theta(1, :) = gamrnd(priors(1), 1 / priors(2), 1, k);
%     Pi = zeros(Iterations, k);
%     Pi(1, :) = rdirichlet(1, ones(1, k));
%     z = zeros(Iterations, length(dat));
% 
%     for i = 2:Iterations
%         z(i, :) = cp_z(dat, Pi(i - 1, :), theta(i - 1, :));
%         for j = 1:k
%             relevant_data = dat(z(i, :) == j);
%             theta(i, j) = cp_theta(relevant_data, priors);
%         end
%         Pi(i, :) = cp_Pi(dat, z(i, :), k);
%     end
% 
%     result.theta = theta;
%     result.weights = Pi;
%     result.occ = z;
% end
% 







% First, create some fake data from a mixture of two exponential distributions
fakeData = [exprnd(1, 1, 50), exprnd(100, 1, 150)];

% Now let's run our Gibbs sampler for our Exponential Mixture Model
Result = ExpMix_Gibbs(fakeData, 500, [1, 1], 2);

Result



figure;

% Subplot 1 and 2
subplot(4, 2, [1, 2]);
plot(Result.theta(:,1), 'Color', [0, 0, 0.85], 'LineWidth', 1.5);
hold on;
plot(Result.theta(:,2), 'Color', [0, 0, 0.85], 'LineWidth', 1.5);
xlabel('MCMC Iteration');
ylabel('\theta_1, \theta_2');
set(gca, 'FontSize', 12);
legend('\theta_1', '\theta_2');
hold off;

% Subplot 3
subplot(4, 2, 3);
histogram(Result.theta(50:end,1), 'Normalization', 'probability', 'EdgeColor', 'none');
xlabel('\theta_1');
ylabel('Posterior Probability');
set(gca, 'FontSize', 12);
hold on;
line([0.01, 0.01], ylim, 'LineWidth', 4, 'Color', [0.21, 0.56, 0.75]);
hold off;

% Subplot 4
subplot(4, 2, 4);
histogram(1./Result.theta(50:end,2), 'Normalization', 'probability', 'EdgeColor', 'none');
xlabel('\theta_2');
ylabel('Posterior Probability');
set(gca, 'FontSize', 12);
hold on;
line([1, 1], ylim, 'LineWidth', 4, 'Color', [0.21, 0.56, 0.75]);
hold off;

% Subplot 5
subplot(4, 2, 5);
plot(Result.weights(:,1), 'Color', [0, 0, 0.85], 'LineWidth', 1.5);
xlabel('MCMC Iteration');
ylabel('w_1');
set(gca, 'FontSize', 12);

% Subplot 6
subplot(4, 2, 6);
plot(Result.weights(:,2), 'Color', [0, 0, 0.85], 'LineWidth', 1.5);
xlabel('MCMC Iteration');
ylabel('w_2');
set(gca, 'FontSize', 12);

% Subplot 7
subplot(4, 2, 7);
histogram(Result.weights(50:end,1), 'Normalization', 'probability', 'EdgeColor', 'none');
xlabel('w_1');
ylabel('Posterior Probability');
set(gca, 'FontSize', 12);
hold on;
line([0.75, 0.75], ylim, 'LineWidth', 4, 'Color', [0.21, 0.56, 0.75]);
hold off;

% Subplot 8
subplot(4, 2, 8);
histogram(Result.weights(50:end,2), 'Normalization', 'probability', 'EdgeColor', 'none');
xlabel('w_2');
ylabel('Posterior Probability');
set(gca, 'FontSize', 12);
hold on;
line([0.25, 0.25], ylim, 'LineWidth', 4, 'Color', [0.21, 0.56, 0.75]);
hold off;


function [Results] = ExpMix_Gibbs(dat, Iterations, pr, k)
    % Function for iteratively sampling from the relevant conditional posteriors
    % Arguments: 
    %   - data vector
    %   - desired number of iterations
    %   - hyperparameters for priors
    %   - number of components
    % Values: A list of Results - includes all posterior samples for Thetas, Pi, and z
  
    % Hyperparameters
    priors = pr;
  
    % Initialize parameters
    theta = NaN(Iterations, k);
    theta(1,:) = gamrnd(priors(1), 1/priors(2), 1, k);
  
    Pi = NaN(Iterations, k);
    Pi(1,:) = rdirichlet(1, ones(1, k));
  
    z = NaN(Iterations, length(dat));
  
    % Main loop
    for i = 2:Iterations
        % Sample through length(data), draw latent indicator variables z	
        z(i,:) = cp_z(dat, Pi(i-1,:), theta(i-1,:));
        
        % Iterate through components j
        for j = 1:k  
            Relevant_Data = dat(z(i,:)==j);
            
            % Sample theta for each cluster
            theta(i,j) = cp_theta(Relevant_Data, priors);		
        end   
        
        % Sample the mixture weights Pi
        Pi(i,:) = cp_Pi(dat, z(i,:), k);
    end
    
    Results.theta = theta;
    Results.weights = Pi;
    Results.occ = z;
end

function theta = cp_theta(dat, priors)
    % Function for drawing from the conditional posterior of the scale parameter Theta
    % Arguments: a vector of data, a vector of hyperparameters for the prior
    % Value: a single draw from the conditional posterior
    
    if isempty(dat)
        A = priors(1);
        B = priors(2);
    else
        A = priors(1) + length(dat);
        B = priors(2) + sum(dat);
    end
    theta = gamrnd(A, 1/B);
end

function Pi = cp_Pi(dat, z, k)
    % Function for drawing from the conditional posterior of the mixture weights
    % Arguments: a data vector, latent indicator variables, number of components
    % Value: a sample of the mixture weights
    
    N_in_each = arrayfun(@(l) length(dat(z==l)), 1:k);
    Pi = rdirichlet(1, N_in_each);
end

function z = cp_z(dat, Pi, theta)
    % Function for drawing from the conditional posterior of latent variables z
    % Arguments: a data vector, mixture weights, scale parameters
    % Value: a length(dat) vector of sampled latent variables
    
    z = zeros(1, length(dat));
    for t = 1:length(dat)
        probs = Pi .* exppdf(dat(t), 1./theta);
        probs = probs / sum(probs);
        [~, ind] = max(mnrnd(1, probs));
        z(t) = ind;
    end
end

function x = rdirichlet(n, alpha)
    % Function for drawing Dirichlet random variables
    % Arguments: the number of desired samples, the Dirichlet parameter vector
    % Value: a draw from the resulting Dirichlet distribution
    
    l = length(alpha);
    x = gamrnd(repmat(alpha, n, 1), 1, n, l);
    sm = sum(x, 2);
    x = x ./ sm;
end



