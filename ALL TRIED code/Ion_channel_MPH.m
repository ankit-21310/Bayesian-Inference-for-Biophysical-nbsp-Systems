

clear 
close all
clc


% Simulate a curve and some noise
V = -150:150;
SyntheticData = BoltzmannCurve(V, [50, 0.05]) + normrnd(0, 0.03, size(V));

% Use this data for parameter inference
Result = MH(SyntheticData, 1000, [40, 0.03, 0.05], [1, 0.002, 0.005]);

% Plotting
subplot(3, 2, 1);
plot(V, SyntheticData);
xlabel('Voltage (mV)');
ylabel('Activation');
title('Synthetic Data with noise N(0,0.03) ');

subplot(3, 2, 2);
plot(Result.a);
xlabel('MCMC Iteration');
ylabel('a');
title('Parameter a ');

subplot(3, 2, 3);
histogram(Result.a(201:end));
xlabel('a (mV)');
ylabel('Counts');
title('Histogram of a');

subplot(3, 2, 4);
plot(Result.b);
xlabel('MCMC Iteration');
ylabel('b (mV)');
title('Parameter b');

subplot(3, 2, 5);
histogram(Result.b(201:end));
xlabel('b');
ylabel('Counts');
title('Histogram of b');



% Function for generating Boltzmann function
function y = BoltzmannCurve(V, parameters)
    % Arguments:
    %   - A vector of the independent variable (V)
    %   - The desired model parameters
    % Value: A vector of the expected model evaluated at the independent variable
    y = 1 ./ (1 + exp((parameters(1) - V) * parameters(2)));
end

% Function for Metropolis-Hastings random walk sampling
function Result = MH(fakeData, iterations, initParams, trans)
    % Arguments:
    %   - A vector of fake noisy data generated from some Boltzmann curve
    %   - Desired number of iterations
    %   - Initial positions of the model parameters
    %   - Step sizes of the random walk transition kernel
    % Value: A struct of results containing all posterior samples for parameters 'a', 'b', and 'sigma'
    
    % Independent variable
    V = -150:150;
    
    % Setting up vague priors
    a_prior = [0, 100];     % mean and std of a normal distribution
    b_prior = [0, 5];       % mean and std of a normal distribution
    sigma_prior = 50;       % scale parameter of an exponential distribution
    
    % Initialize parameters
    a = NaN(1, iterations);
    a(1) = initParams(1);
    b = NaN(1, iterations);
    b(1) = initParams(2);
    sigma = NaN(1, iterations);
    sigma(1) = initParams(3);
    
    % Main loop
    for i = 2:iterations
        % Sample parameter 'a'
        old_expected_curve = BoltzmannCurve(V, [a(i-1), b(i-1)]);
        
        % Random walk for proposal
        a_proposal = a(i-1) + normrnd(0, trans(1));
        proposal_expected_curve = BoltzmannCurve(V, [a_proposal, b(i-1)]);
        
        % Calculate posterior probabilities for proposal point and previous point
        Old_LogPosterior = sum(log(normpdf(fakeData, old_expected_curve, sigma(i-1)))) + log(normpdf(a(i-1), a_prior(1), a_prior(2)));
        Proposal_LogPosterior = sum(log(normpdf(fakeData, proposal_expected_curve, sigma(i-1)))) + log(normpdf(a_proposal, a_prior(1), a_prior(2)));
        
        % Compare Old Posterior and Proposal Posterior, and do accept/reject
        if rand(1) < exp(Proposal_LogPosterior - Old_LogPosterior)
            a(i) = a_proposal;
        else
            a(i) = a(i-1);
        end
        
        % Sample parameter 'b'
        old_expected_curve = BoltzmannCurve(V, [a(i), b(i-1)]);
        
        % Random walk for proposal
        b_proposal = b(i-1) + normrnd(0, trans(2));
        proposal_expected_curve = BoltzmannCurve(V, [a(i), b_proposal]);
        
        % Calculate posterior probabilities for proposal point and previous point
        Old_LogPosterior = sum(log(normpdf(fakeData, old_expected_curve, sigma(i-1)))) + log(normpdf(b(i-1), b_prior(1), b_prior(2)));
        Proposal_LogPosterior = sum(log(normpdf(fakeData, proposal_expected_curve, sigma(i-1)))) + log(normpdf(b_proposal, b_prior(1), b_prior(2)));
        
        % Compare Old Posterior and Proposal Posterior, and do accept/reject
        if rand(1) < exp(Proposal_LogPosterior - Old_LogPosterior)
            b(i) = b_proposal;
        else
            b(i) = b(i-1);
        end
        
        % Sample parameter 'sigma'
        old_expected_curve = BoltzmannCurve(V, [a(i), b(i)]);
        
        % Random walk for proposal
        sigma_proposal = sigma(i-1) + normrnd(0, trans(3));
        proposal_expected_curve = BoltzmannCurve(V, [a(i), b(i)]); % the expectations are actually the same in this case, but the Posteriors will differ due to different levels of noise (sigma) in the likelihoods
        
        % Calculate posterior probabilities for proposal point and previous point
        Old_LogPosterior = sum(log(normpdf(fakeData, old_expected_curve, sigma(i-1)))) + log(exppdf(sigma(i-1), sigma_prior));
        Proposal_LogPosterior = sum(log(normpdf(fakeData, proposal_expected_curve, sigma_proposal))) + log(exppdf(sigma_proposal, sigma_prior));
        
        % Compare Old Posterior and Proposal Posterior, and do accept/reject
        if rand(1) < exp(Proposal_LogPosterior - Old_LogPosterior)
            sigma(i) = sigma_proposal;
        else
            sigma(i) = sigma(i-1);
        end
    end
    
    Result = struct('a', a, 'b', b, 'sigma', sigma);
end