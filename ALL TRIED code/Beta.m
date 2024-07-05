clear
close all
clc 


% Define theta values
theta = 0:0.01:1;

% beta pdf
beta_pdf = @(th, a, b) exp(gammaln(a + b) - gammaln(a) - gammaln(b) + (a - 1) * log(th) + (b - 1) * log(1 - th));

b1 = beta_pdf(theta,1,1);
b2 = beta_pdf(theta,5,5);
b3 = beta_pdf(theta,3,10);  


% Plot beta distributions
plot(theta, b1, 'k-', 'LineWidth', 5, 'Color', [0, 0, 0, 0.44]);
hold on;
plot(theta, b2, 'b-', 'LineWidth', 5);
plot(theta, b3, 'Color', [0.498, 1, 0.831], 'LineWidth', 5);
hold off;

% Set axis limits and labels
ylim([0, 5]);
xlabel('\theta', 'FontSize', 13);
ylabel('p(\theta)', 'FontSize', 13);

% Add labels for distributions
text(0.9, 1.25, '(1,1)', 'FontSize', 13);
text(0.5, 2.6, '(10,3)', 'FontSize', 13);
text(0.2, 3.8, '(3,10)', 'FontSize', 13);

% Adjust margins
set(gca, 'Fontsize', 12);


