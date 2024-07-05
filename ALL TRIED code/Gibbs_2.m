% % % 
% % % 
% % % % Generate fake data from a mixture of two exponential distributions
% % % lambda1 = 1;
% % % lambda2 = 0.01;
% % % fakeData = [my_exprnd(lambda1, [1, 50]), my_exprnd(lambda2, [1, 150])];
% % % 
% % % % Now let's run our Gibbs sampler for our Exponential Mixture Model
% % % Result = ExpMix_Gibbs(fakeData, 500, [1, 1], 2);
% % % 
% % % % Plotting the results
% % % figure;
% % % mixturePlot(fakeData, Result, 'ylab', 'Counts', 'xlab', 'Data (Log Scale)', 'bns', 20);
% % % 
% % % 
% % % 
% % % % Function for drawing Dirichlet random variables
% % % function x = rdirichlet(n, alpha)
% % %     % ARGUMENTS: the number of desired samples, the Dirichlet parameter vector
% % %     % VALUE: a draw from the resulting Dirichlet distribution
% % %     l = length(alpha);
% % %     x = gamrnd(ones(n, l), repmat(alpha, n, 1));
% % %     sm = sum(x, 2);
% % %     x = x ./ repmat(sm, 1, l);
% % % end
% % % 
% % % % Function for drawing from the conditional posterior of the scale parameter Theta
% % % function theta = cp_theta(dat, priors)
% % %     % ARGUMENTS: a vector of data, a vector of hyperparameters for the prior
% % %     % VALUE: a single draw from the conditional posterior
% % %     if isempty(dat)
% % %         A = priors(1);
% % %         B = priors(2);
% % %     else
% % %         A = priors(1) + length(dat);
% % %         B = priors(2) + sum(dat);
% % %     end
% % %     theta = gamrnd(A, 1/B);
% % % end
% % % 
% % % % Function for drawing from the conditional posterior of the mixture weights 
% % % function Pi = cp_Pi(dat, z, k)
% % %     % ARGUMENTS: a data vector, latent indicator variables, number of components
% % %     % VALUE: a sample of the mixture weights
% % %     N_in_each = arrayfun(@(l) sum(dat(z==l)), 1:k);
% % %     Pi = rdirichlet(1, N_in_each);
% % % end
% % % 
% % % % Function for drawing from the conditional posterior of latent variables z
% % % function z = cp_z(dat, Pi, theta)
% % %     % ARGUMENTS: a data vector, mixture weights, scale parameters
% % %     % VALUE: a length(dat) vector of sampled latent variables
% % %     probs = zeros(length(dat), length(theta));
% % %     for t = 1:length(dat)
% % %         for l = 1:length(theta)
% % %             probs(t, l) = Pi(l) * exppdf(dat(t), 1/theta(l));
% % %         end
% % %         probs(t, :) = probs(t, :) / sum(probs(t, :));
% % %         z(t) = find(mnrnd(1, probs(t, :)) == 1);
% % %     end
% % % end
% % % 
% % % % Function for iteratively sampling from the relevant conditional posteriors
% % % function Results = ExpMix_Gibbs(dat, Iterations, pr, k)
% % %     % ARGUMENTS: 
% % %     %   - data vector
% % %     %   - desired number of iterations
% % %     %   - hyperparameters for priors
% % %     %   - number of components
% % %     % VALUES: A struct of Results - includes all posterior samples for Thetas, Pi, and z
% % % 
% % %     % hyperparameters
% % %     priors = pr;
% % % 
% % %     % Initialize parameters
% % %     theta = NaN(Iterations, k);
% % %     theta(1, :) = gamrnd(priors(1), 1/priors(2), 1, k);
% % % 
% % %     Pi = NaN(Iterations, k);
% % %     Pi(1, :) = rdirichlet(1, ones(1, k));
% % % 
% % %     z = NaN(Iterations, length(dat));
% % % 
% % %     % main loop
% % %     for i = 2:Iterations
% % %         % sample through length(data), draw latent indicator variables z	
% % %         z(i, :) = cp_z(dat, Pi(i-1, :), theta(i-1, :));
% % %         % iterate through components j
% % %         for j = 1:k
% % %             Relevant_Data = dat(z(i, :) == j);
% % %             % sample theta for each cluster
% % %             theta(i, j) = cp_theta(Relevant_Data, priors);		
% % %         end   
% % %         % sample the mixture weights Pi
% % %         Pi(i, :) = cp_Pi(dat, z(i, :), k); 
% % %     end
% % %     Results.theta = theta;
% % %     Results.weights = Pi;
% % %     Results.occ = z;
% % % end
% % % 
% % % % Function to generate exponential random numbers from a uniform distribution
% % % function x = my_exprnd(lambda, sz)
% % %     u = rand(sz);
% % %     x = -log(1 - u) / lambda;
% % % end
% % % 
% % % % Function for generating gamma-distributed random numbers using the Marsaglia-Tsang method
% % % function x = gamrnd(shape, scale, sz)
% % %     % Generate gamma-distributed random numbers using Marsaglia-Tsang method
% % %     % shape: shape parameter of the gamma distribution
% % %     % scale: scale parameter of the gamma distribution
% % %     % sz: size of the output array
% % % 
% % %     if nargin < 3
% % %         sz = 1;
% % %     end
% % % 
% % %     x = zeros(sz);
% % %     for i = 1:shape
% % %         u = rand(sz);
% % %         x = x - log(u);
% % %     end
% % %     x = x * scale;
% % % end
% % % 
% % % % % Function for generating gamma-distributed random numbers
% % % % function x = gamrnd(shape, scale, varargin)
% % % %     x = random('Gamma', shape, scale, varargin{:});
% % % % end
% % 
% % 
% % % Generate fake data from a mixture of two exponential distributions
% % lambda1 = 1;
% % lambda2 = 0.01;
% % fakeData = [my_exprnd(lambda1, [1, 50]), my_exprnd(lambda2, [1, 150])];
% % 
% % % Now let's run our Gibbs sampler for our Exponential Mixture Model
% % Result = ExpMix_Gibbs(fakeData, 500, [1, 1], 2);
% % 
% % % Plotting the results
% % figure;
% % mixturePlot(fakeData, Result, 'ylab', 'Counts', 'xlab', 'Data (Log Scale)', 'bns', 20);
% % 
% % % Function for drawing Dirichlet random variables
% % function x = rdirichlet(n, alpha)
% %     % ARGUMENTS: the number of desired samples, the Dirichlet parameter vector
% %     % VALUE: a draw from the resulting Dirichlet distribution
% %     l = length(alpha);
% %     x = gamrnd(ones(n, l), repmat(alpha, n, 1));
% %     sm = sum(x, 2);
% %     x = x ./ repmat(sm, 1, l);
% % end
% % 
% % % Function for drawing from the conditional posterior of the scale parameter Theta
% % function theta = cp_theta(dat, priors)
% %     % ARGUMENTS: a vector of data, a vector of hyperparameters for the prior
% %     % VALUE: a single draw from the conditional posterior
% %     if isempty(dat)
% %         A = priors(1);
% %         B = priors(2);
% %     else
% %         A = priors(1) + length(dat);
% %         B = priors(2) + sum(dat);
% %     end
% %     theta = my_gamrnd(A, 1/B);
% % end
% % 
% % % Function for drawing from the conditional posterior of the mixture weights 
% % function Pi = cp_Pi(dat, z, k)
% %     % ARGUMENTS: a data vector, latent indicator variables, number of components
% %     % VALUE: a sample of the mixture weights
% %     N_in_each = arrayfun(@(l) sum(dat(z==l)), 1:k);
% %     Pi = rdirichlet(1, N_in_each);
% % end
% % 
% % % Function for drawing from the conditional posterior of latent variables z
% % function z = cp_z(dat, Pi, theta)
% %     % ARGUMENTS: a data vector, mixture weights, scale parameters
% %     % VALUE: a length(dat) vector of sampled latent variables
% %     probs = zeros(length(dat), length(theta));
% %     for t = 1:length(dat)
% %         for l = 1:length(theta)
% %             probs(t, l) = Pi(l) * exppdf(dat(t), 1/theta(l));
% %         end
% %         probs(t, :) = probs(t, :) / sum(probs(t, :));
% %         z(t) = find(mnrnd(1, probs(t, :)) == 1);
% %     end
% % end
% % 
% % % Function for iteratively sampling from the relevant conditional posteriors
% % function Results = ExpMix_Gibbs(dat, Iterations, pr, k)
% %     % ARGUMENTS: 
% %     %   - data vector
% %     %   - desired number of iterations
% %     %   - hyperparameters for priors
% %     %   - number of components
% %     % VALUES: A struct of Results - includes all posterior samples for Thetas, Pi, and z
% % 
% %     % hyperparameters
% %     priors = pr;
% % 
% %     % Initialize parameters
% %     theta = NaN(Iterations, k);
% % 
% %     theta(1, :) = my_gamrnd(priors(1), 1/priors(2));
% % 
% %     Pi = NaN(Iterations, k);
% %     Pi(1, :) = rdirichlet(1, ones(1, k));
% % 
% %     z = NaN(Iterations, length(dat));
% % 
% %     % main loop
% %     for i = 2:Iterations
% %         % sample through length(data), draw latent indicator variables z	
% %         z(i, :) = cp_z(dat, Pi(i-1, :), theta(i-1, :));
% %         % iterate through components j
% %         for j = 1:k
% %             Relevant_Data = dat(z(i, :) == j);
% %             % sample theta for each cluster
% %             theta(i, j) = cp_theta(Relevant_Data, priors);		
% %         end   
% %         % sample the mixture weights Pi
% %         Pi(i, :) = cp_Pi(dat, z(i, :), k); 
% %     end
% %     Results.theta = theta;
% %     Results.weights = Pi;
% %     Results.occ = z;
% % end
% % 
% % % Function to generate exponential random numbers from a uniform distribution
% % function x = my_exprnd(lambda, sz)
% %     u = rand(sz);
% %     x = -log(1 - u) / lambda;
% % end
% % 
% % % Function for generating gamma-distributed random numbers using the Marsaglia-Tsang method
% % % Function for generating gamma-distributed random numbers using the Marsaglia-Tsang method
% % function x = my_gamrnd(shape, scale)
% %     % Generate gamma-distributed random numbers using Marsaglia-Tsang method
% %     % shape: shape parameter of the gamma distribution
% %     % scale: scale parameter of the gamma distribution
% % 
% %     u = rand(shape, 1);
% %     x = zeros(size(u));
% %     for i = 1:shape
% %         x(i) = -log(u(i));
% %     end
% %     x = x * scale;
% % end
% 







% First, create some fake data from a mixture of two exponential distributions
rng('default'); % For reproducibility
fakeData = [exprnd(1, 50, 1); exprnd(0.01, 150, 1)];

% Now let's run our Gibbs sampler for our Exponential Mixture Model
Iterations = 500;
pr = [1, 1]; % hyperparameters for priors
k = 2; % number of components
Result = ExpMix_Gibbs(fakeData, Iterations, pr, k);

% Assuming Result is the struct obtained from the Gibbs sampling
end_index = size(Result.theta, 1);
figure;
subplot(4, 2, 1);
plot(Result.theta(:, 1), 'Color', [0 0 0.85]);
ylabel('\theta_1', 'Interpreter', 'tex');
xlabel('MCMC Iteration');
subplot(4, 2, 2);
plot(Result.theta(:, 2), 'Color', [0 0 0.85]);
ylabel('\theta_2', 'Interpreter', 'tex');
xlabel('MCMC Iteration');
subplot(4, 2, 3);
histogram(Result.theta(50:end_index, 1), 'Normalization', 'probability');
xlabel('\theta_1', 'Interpreter', 'tex');
ylabel('Posterior Probability');
hold on;
plot([0.01 0.01], ylim, 'LineWidth', 4, 'Color', [0.21 0.56 0.75]);
hold off;
subplot(4, 2, 4);
histogram(1./Result.theta(50:end_index, 2), 'Normalization', 'probability');
xlabel('\theta_2', 'Interpreter', 'tex');
ylabel('Posterior Probability');
hold on;
plot([1 1], ylim, 'LineWidth', 4, 'Color', [0.21 0.56 0.75]);
hold off;

subplot(4, 2, 5);
plot(Result.weights(:, 1), 'Color', [0 0 0.85]);
ylabel('w_1', 'Interpreter', 'tex');
xlabel('MCMC Iteration');
subplot(4, 2, 6);
plot(Result.weights(:, 2), 'Color', [0 0 0.85]);
ylabel('w_2', 'Interpreter', 'tex');
xlabel('MCMC Iteration');
subplot(4, 2, 7);
histogram(Result.weights(50:end_index, 1), 'Normalization', 'probability');
xlabel('w_1', 'Interpreter', 'tex');
ylabel('Posterior Probability');
hold on;
plot([0.75 0.75], ylim, 'LineWidth', 4, 'Color', [0.21 0.56 0.75]);
hold off;
subplot(4, 2, 8);
histogram(Result.weights(50:end_index, 2), 'Normalization', 'probability');
xlabel('w_2', 'Interpreter', 'tex');
ylabel('Posterior Probability');
hold on;
plot([0.25 0.25], ylim, 'LineWidth', 4, 'Color', [0.21 0.56 0.75]);
hold off;


% 
function x = rdirichlet(n, alpha)
    % ARGUMENTS: the number of desired samples, the Dirichlet parameter vector
    % VALUE: a draw from the resulting Dirichlet distribution
    l = length(alpha);
    x = gamrnd(repmat(alpha, n, 1), 1, n, l);
    sm = sum(x, 2);
    x = x ./ repmat(sm, 1, l);
end

function theta = cp_theta(dat, priors)
    % ARGUMENTS: a vector of data, a vector of hyperparameters for the prior
    % VALUE: a single draw from the conditional posterior
    if isempty(dat)
        A = priors(1);
        B = priors(2);
    else
        A = (priors(1) + length(dat));
        B = priors(2) + sum(dat);
    end
    theta = gamrnd(A, 1/B);
end

function Pi = cp_Pi(dat, z, k)
    % ARGUMENTS: a data vector, latent indicator variables, number of components
    % VALUE: a sample of the mixture weights
    N_in_each = zeros(1, k);
    for l = 1:k
        N_in_each(l) = length(dat(z == l));
    end
    Pi = rdirichlet(1, N_in_each);
end

function z = cp_z(dat, Pi, theta)
    % ARGUMENTS: a data vector, mixture weights, scale parameters
    % VALUE: a length(dat) vector of sampled latent variables
    n = length(dat);
    k = length(theta);
    z = zeros(1, n);
    for t = 1:n
        probs = zeros(1, k);
        for l = 1:length(theta)
            probs(l) = Pi(l) * exppdf(dat(t), 1/theta(l));
        end
        probs = probs / sum(probs);
        [~, z(t)] = max(mnrnd(1, probs));
    end
end

function Results = ExpMix_Gibbs(dat, Iterations, pr, k)
    % ARGUMENTS: 
    %   - data vector
    %   - desired number of iterations
    %   - hyperparameters for priors
    %   - number of components
    % VALUES: A struct of Results - includes all posterior samples for Thetas, Pi, and z

    % hyperparameters
    priors = pr;

    % Initialize parameters
    theta = NaN(Iterations, k);
    theta(1, :) = gamrnd(priors(1), 1/priors(2), 1, k);

    Pi = NaN(Iterations, k);
    Pi(1, :) = rdirichlet(1, ones(1, k));

    z = NaN(Iterations, length(dat));

    % main loop
    for i = 2:Iterations
        % sample through length(data), draw latent indicator variables z    
        z(i, :) = cp_z(dat, Pi(i-1, :), theta(i-1, :));
        % iterate through components j
        for j = 1:k
            Relevant_Data = dat(z(i, :) == j);
            % sample theta for each cluster
            theta(i, j) = cp_theta(Relevant_Data, priors);
        end
        % sample the mixture weights Pi
        Pi(i, :) = cp_Pi(dat, z(i, :), k);
    end
    Results = struct('theta', theta, 'weights', Pi, 'occ', z);
end


