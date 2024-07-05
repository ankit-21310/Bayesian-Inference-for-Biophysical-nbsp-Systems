
clear
close all
clc

% First, create some fake data from a mixture of two exponential distributions
fakeData = [exprnd(1, 1, 50), exprnd(1/0.01, 1, 150)];

% Now let's run our Gibbs sampler for our Exponential Mixture Model
Result = ExpMix_Gibbs(fakeData, 500, [1, 1], 2);



mixturePlot(fakeData, Result, 'ylab', 'Counts', 'xlab', 'Data (Log Scale)', 'cex_lab', 1.3, 'bns', 20);

figure;
set(gcf, 'Position', get(0, 'Screensize')); % Maximize figure window

% Layout
subplot(4, 2, 1);
subplot(4, 2, 2);
subplot(4, 2, 3);
subplot(4, 2, 4);
subplot(4, 2, 5);
subplot(4, 2, 6);
subplot(4, 2, 7);
subplot(4, 2, 8);

% Plotting Result.theta[,1]
subplot(4, 2, 1);
end_val = size(Result.theta, 1);
plot(Result.theta(:,1), 'Color', [0, 0, 0.85]);
ylabel('\theta_1', 'FontSize', 12);
xlabel('MCMC Iteration', 'FontSize', 12);

% Plotting Result.theta[,2]
subplot(4, 2, 2);
plot(Result.theta(:,2), 'Color', [0, 0, 0.85]);
ylabel('\theta_2', 'FontSize', 12);
xlabel('MCMC Iteration', 'FontSize', 12);

% Plotting histogram of Result.theta[50:end,1]
subplot(4, 2, 3);
histogram(Result.theta(50:end,1), 'Normalization', 'probability');
xlabel('\theta_1', 'FontSize', 12);
ylabel('Posterior Probability', 'FontSize', 12);
hold on;
line([0.01 0.01], ylim, 'LineWidth', 4, 'Color', [0.21, 0.56, 0.75]);

% Plotting histogram of 1/Result.theta[50:end,2]
subplot(4, 2, 4);
histogram(1./Result.theta(50:end,2), 'Normalization', 'probability');
xlabel('\theta_2', 'FontSize', 12);
ylabel('Posterior Probability', 'FontSize', 12);
hold on;
line([1 1], ylim, 'LineWidth', 4, 'Color', [0.21, 0.56, 0.75]);

% Plotting Result.weights[,1]
subplot(4, 2, 5);
plot(Result.weights(:,1), 'Color', [0, 0, 0.85]);
ylabel('w_1', 'FontSize', 12);
xlabel('MCMC Iteration', 'FontSize', 12);

% Plotting Result.weights[,2]
subplot(4, 2, 6);
plot(Result.weights(:,2), 'Color', [0, 0, 0.85]);
ylabel('w_2', 'FontSize', 12);
xlabel('MCMC Iteration', 'FontSize', 12);

% Plotting histogram of Result.weights[50:end,1]
subplot(4, 2, 7);
histogram(Result.weights(50:end,1), 'Normalization', 'probability');
xlabel('w_1', 'FontSize', 12);
ylabel('Posterior Probability', 'FontSize', 12);
hold on;
line([0.75 0.75], ylim, 'LineWidth', 4, 'Color', [0.21, 0.56, 0.75]);

% Plotting histogram of Result.weights[50:end,2]
subplot(4, 2, 8);
histogram(Result.weights(50:end,2), 'Normalization', 'probability');
xlabel('w_2', 'FontSize', 12);
ylabel('Posterior Probability', 'FontSize', 12);
hold on;
line([0.25 0.25], ylim, 'LineWidth', 4, 'Color', [0.21, 0.56, 0.75]);










function x = rdirichlet(n, alpha)
    % ARGUMENTS: the number of desired samples, the Dirichlet parameter vector
    % VALUE: a draw from the resulting Dirichlet distribution
    l = length(alpha);
    x = zeros(n, l);
    for i = 1:n
        gamma_samples = gamrnd(alpha, 1);
        x(i, :) = gamma_samples / sum(gamma_samples);
    end
end

function theta = cp_theta(dat, priors)
    % ARGUMENTS: a vector of data, a vector of hyperparameters for the prior
    % VALUE: a single draw from the conditional posterior
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
    z = zeros(1, length(dat));
    for t = 1:length(dat)
        probs = zeros(1, length(theta));
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
    % VALUES: A list of Results - includes all posterior samples for Thetas, Pi, and z
  
    %hyperparameters
    priors = pr;
    
    % Initialize parameters
    theta = NaN(Iterations, k);
    theta(1, :) = gamrnd(priors(1), 1/priors(2), [1, k]);
    
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
    Results.theta = theta;
    Results.weights = Pi;
    Results.occ = z;
end


function mixturePlot(rawData, analysis, varargin)
    COLS = [0.118, 0.565, 1.000;   % dodgerblue
            0.698, 0.133, 0.133;   % firebrick
            0.498, 1.000, 0.831;   % aquamarine3
            0.169, 0.451, 0.604;   % #2B739A
            1.000, 0.522, 0.678;   % #FF85AD
            0.000, 0.000, 0.000];  % black
    
    bns = 25;
    if nargin > 2
        for i = 1:2:length(varargin)
            if strcmpi(varargin{i}, 'bns')
                bns = varargin{i+1};
            end
        end
    end
    
    br = logspace(log10(min(rawData)), log10(max(rawData)), bns);
    h = histcounts(rawData, br);
    mids = exp((log(br(1:end-1)) + log(br(2:end))) / 2);
    
    figure;
    semilogx(mids, h, 'LineWidth', 2, 'Color', [0, 0, 0.8, 0.5]);
    hold on;
    
    x = logspace(log10(0.000001), log10(100000), 200);
    S = zeros(size(x));
    counter = 1;
    for i = unique(analysis.occ(end, :))
        d = rawData(analysis.occ(end, :) == i);
        scatter(d, zeros(size(d)), 50, COLS(counter, :), 'filled', 'MarkerFaceAlpha', 0.5);
        
        tau = analysis.theta(end, i);
        w = analysis.weights(end, i);
        g = exp(log(x) - log(1 / tau) - exp(log(x) - log(1 / tau)));
        g = g / max(g);
        g = g * max(h) * w;
        plot(x, g, 'LineWidth', 3, 'Color', COLS(counter, :));
        S = S + g;
        counter = counter + 1;
    end
    
    plot(x, S, 'LineWidth', 8, 'Color', [0, 0, 0, 0.2]);
    
    xlabel('Data (Log Scale)', 'FontSize', 12);
    ylabel('Counts', 'FontSize', 12);
    set(gca, 'FontSize', 12);
    xlim([min(rawData), max(rawData)]);
    xticks([0.05, 0.50, 5.00, 50.00, 500.00]);
    xticklabels({'0.05', '0.50', '5.00', '50.00', '500.00'});
end


