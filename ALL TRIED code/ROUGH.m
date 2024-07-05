% % % Clear workspace, close all figures, and clear command window
clear;
close all;
clc;

% % % Set the random seed for reproducibility
% % rng(42);
% % 
% % % Lotka-Volterra ODE system
% % lv_ode = @(t, y, a, b) [a * y(1) - y(1) * y(2); b * y(1) * y(2) - y(2)];
% % 
% % % ABC SMC parameters
% % T = 5; % Number of populations
% % N = 1000; % Number of particles in each population
% % epsilon = [30.0 16.0 6.0 5.0 4.3]; % Tolerance levels for each population
% % prior_min = -10; % Prior distribution boundaries
% % prior_max = 10; % Uniform prior [-10, 10]
% % perturb_variance = 0.1; % Variance for the perturbation kernel (uniform)
% % 
% % % Observed data from the Lotka-Volterra model
% % a = 1; % True parameter 'a'
% % b = 1; % True parameter 'b'
% % x0 = [1.0, 0.5]; % Initial conditions
% % tspan = [0, 15]; % Time span for solving the ODE
% % sample_times = linspace(0, 15, 8); % Times to sample data
% % 
% % % Simulate the deterministic Lotka-Volterra system
% % [t, Y] = ode15s(@(t, Y) lv_ode(t, Y, a, b), tspan, x0);
% % 
% % % Generate noisy data for prey and predator
% % prey_data = interp1(t, Y(:, 1), sample_times) + randn(1, 8) * 0.5; % Noisy prey data
% % predator_data = interp1(t, Y(:, 2), sample_times) + randn(1, 8) * 0.5; % Noisy predator data
% % 
% % observed_data = {prey_data, predator_data}; % Observed noisy data
% % 
% % % Distance function to calculate squared differences
% % calc_distance = @(x, y, x_d, y_d) sum((x - x_d).^2) + sum((y - y_d).^2);
% % 
% % % Initialize results for storing particles and weights
% % res = zeros(N, 3); % Results matrix (a, b, distance)
% % weights = zeros(T, N); % Weights for each population
% % thetas = zeros(T, N, 2); % Particles (each element is a 2-element vector)
% % data_generation_steps = zeros(1, T); % Data generation steps for each population
% % 
% % 
% % % ABC SMC algorithm
% % for t = 1:T
% % 
% %     i = 1; % Counter for accepted particles
% %     j = 0; % Counter for total proposals
% % 
% %     while i <= N
% %         if t == 1
% %             % For the first population, sample directly from the prior
% %             theta_s_s = [unifrnd(prior_min, prior_max), unifrnd(prior_max, prior_max)];
% %         else
% %             % Resample from the previous population with weights
% %             prev_idx = randsample(N, 1, true, weights(t - 1, :));
% %             prev_theta = thetas(t - 1, prev_idx, :);
% % 
% %             % Perturbation using a uniform kernel with a specified variance
% %             theta_s_s = prev_theta + unifrnd(-perturb_variance, perturb_variance, [1, 2]);
% %         end
% % 
% %         % Simulate data with proposed parameters
% %         [t_, Y] = ode15s(@(t_, Y) lv_ode(t_, Y, theta_s_s(1), theta_s_s(2)), tspan, x0);
% %         simulated_prey_data = interp1(t_, Y(:, 1), sample_times);
% %         simulated_predator_data = interp1(t_, Y(:, 2), sample_times);
% % 
% % 
% %         % Calculate the distance between simulated and observed data
% %         distance = calc_distance(simulated_prey_data, simulated_predator_data, prey_data, predator_data);
% % 
% %         if distance < epsilon(t)
% %         % If the distance is within the tolerance level, accept the particle
% % 
% %         % Store accepted particle and its weight
% %         thetas(t, i, :) = theta_s_s;
% % 
% %         if t == 1
% %             % Weight for the first population is uniform (1)
% %             weights(t, i) = 1;
% %         else
% %             % Weight for subsequent populations
% %             normalization = sum(weights(t - 1, :) .* uniformpdf(thetas(t - 1, :, :), theta_s_s, perturb_variance));
% %             weights(t, i) = normpdf(theta_s_s, [prior_min prior_min], [prior_max prior_max]) / normalization;
% %         end
% % 
% %         % Increment accepted particle counter
% %         i = i + 1;
% %         end
% % 
% %         % Increment proposal counter
% %         j = j + 1;
% %     end
% % 
% %     % Normalize weights for this population
% %     weights(t, :) = weights(t, :) / sum(weights(t, :));
% % 
% %     % Store the number of proposals for this population
% %     data_generation_steps(t) = j;
% % end
% % 
% % % Display results for accepted particles
% % disp("Data Generation Steps:");
% % disp(data_generation_steps);
% % 
% % % Plot the histograms of the accepted parameters
% % figure;
% % subplot(1, 2, 1);
% % histogram(thetas(end, :, 1));
% % xlabel('Parameter a');
% % ylabel('Frequency');
% % title('Histogram of Parameter a in the Last Population');
% % 
% % subplot(1, 2, 2);
% % histogram(thetas(end, :, 2));
% % xlabel('Parameter b');
% % ylabel('Frequency');
% % title('Histogram of Parameter b in the Last Population');






% Now let's put all that code together and try Gibbs sampling

% First, create some fake data from a mixture of two exponential distributions
fakeData = [exprnd(1, 1, 50), exprnd(0.01, 1, 150)];

% Now let's run our Gibbs sampler for our Exponential Mixture Model
Result = ExpMix_Gibbs(fakeData, 500, [1, 1], 2);




% Define layout for subplots
figure;
subplot(4, 2, 1);
plot(Result.theta(:, 1), 'Color', [0, 0, 0.85], 'LineWidth', 1.5); % Adjust color using RGB values
ylabel('\theta_1');
xlabel('MCMC Iteration');
title('\theta_1 Evolution');

subplot(4, 2, 2);
plot(Result.theta(:, 2), 'Color', [0, 0, 0.85], 'LineWidth', 1.5); % Adjust color using RGB values
ylabel('\theta_2');
xlabel('MCMC Iteration');
title('\theta_2 Evolution');

subplot(4, 2, 3);
histogram(Result.theta(50:end, 1), 'Normalization', 'probability', 'FaceColor', [0.21, 0.56, 0.75]);
xlabel('\theta_1');
ylabel('Probability Density');
title('\theta_1 Posterior');

subplot(4, 2, 4);
histogram(1./Result.theta(50:end, 2), 'Normalization', 'probability', 'FaceColor', [0.21, 0.56, 0.75]);
xlabel('\theta_2');
ylabel('Probability Density');
title('\theta_2 Posterior');

subplot(4, 2, 5);
plot(Result.weights(:, 1), 'Color', [0, 0, 0.85], 'LineWidth', 1.5); % Adjust color using RGB values
ylabel('w_1');
xlabel('MCMC Iteration');
title('w_1 Evolution');

subplot(4, 2, 6);
plot(Result.weights(:, 2), 'Color', [0, 0, 0.85], 'LineWidth', 1.5); % Adjust color using RGB values
ylabel('w_2');
xlabel('MCMC Iteration');
title('w_2 Evolution');

subplot(4, 2, 7);
histogram(Result.weights(50:end, 1), 'Normalization', 'probability', 'FaceColor', [0.21, 0.56, 0.75]);
xlabel('w_1');
ylabel('Probability Density');
title('w_1 Posterior');

subplot(4, 2, 8);
histogram(Result.weights(50:end, 2), 'Normalization', 'probability', 'FaceColor', [0.21, 0.56, 0.75]);
xlabel('w_2');
ylabel('Probability Density');
title('w_2 Posterior');













% Function for drawing Dirichlet random variables
function x = rdirichlet(n, alpha)
    % Arguments: number of desired samples, Dirichlet parameter vector
    % Value: a draw from the resulting Dirichlet distribution
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
    % Arguments: a vector of data, a vector of hyperparameters for the prior
    % Value: a single draw from the conditional posterior
    A = priors(1) + length(dat);
    B = priors(2) + sum(dat);
    theta = gamrnd(A, 1 / B);
end

% Function for drawing from the conditional posterior of the mixture weights
function Pi = cp_Pi(dat, z, k)
    % Arguments: a data vector, latent indicator variables, number of components
    % Value: a sample of the mixture weights
    N_in_each = zeros(1, k);
    for l = 1:k
        N_in_each(l) = sum(dat(z == l));
    end
    Pi = rdirichlet(1, N_in_each);
end

% Function for drawing from the conditional posterior of latent variables z
function z = cp_z(dat, Pi, theta)
    % Arguments: a data vector, mixture weights, scale parameters
    % Value: a length(dat) vector of sampled latent variables
    n = length(dat);
    z = zeros(1, n);
    for t = 1:n
        probs = Pi .* exppdf(dat(t), 1 ./ theta);
        probs = probs / sum(probs);
        z(t) = find(mnrnd(1, probs));
    end
end

% Function for iteratively sampling from the relevant conditional posteriors
function Results = ExpMix_Gibbs(dat, Iterations, pr, k)
    % Arguments:
    %   - data vector
    %   - desired number of iterations
    %   - hyperparameters for priors
    %   - number of components
    % Value: A list of Results - includes all posterior samples for Thetas, Pi, and z
    
    % Initialize parameters
    priors = pr;
    theta = zeros(Iterations, k);
    theta(1, :) = gamrnd(priors(1), 1 / priors(2), 1, k);
    Pi = zeros(Iterations, k);
    Pi(1, :) = rdirichlet(1, ones(1, k));
    z = zeros(Iterations, length(dat));
    
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


