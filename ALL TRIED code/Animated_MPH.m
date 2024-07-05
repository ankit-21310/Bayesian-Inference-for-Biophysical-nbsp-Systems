clear 
close all
clc


% Set seed for reproducibility
rng(0);
steps = 3000;
step_size = 0.4;

samples = metropolis_sample(@mixture_of_gaussians_pdf, steps, step_size, 0);

% Prepare static plot
edges = linspace(min(samples), max(samples), 100);
[counts, edges] = histcounts(samples, edges);
bins = discretize(samples, edges);
counter = zeros(size(bins));
for idx = 1:numel(bins)
    counter(idx) = sum(bins(1:idx) == bins(idx));
end
counter = counter / max(counter);



% Plot static graph
fig = figure;
ax = axes;
hold(ax, 'on');
ylim(ax, [0, 1]);
xlim(ax, [min(bins), max(bins)]);
ax.YAxis.Visible = 'off';
ax.XAxis.Visible = 'off';
cmap = parula(length(bins));
scatter(ax, bins, counter, [], cmap, 'filled');
hold(ax, 'off');

fig2 = figure;
x = linspace(-6, 6, 1000);
% Define the function
y = 0.5 * normpdf(x, -2, 1) + 0.5 * normpdf(x, 2, 1);


subplot(2, 1, 1);
plot(samples, 'LineWidth', 1.5);
title('Trace Plot');
xlabel('Iteration');
ylabel('Sample Value');
grid on;

subplot(2, 1, 2);
plot(x, y, 'LineWidth', 2);
title('Target distribution: 0.5 * normpdf(x, -2, 1) + 0.5 * normpdf(x, 2, 1)');
xlabel('x');
ylabel('Probability Density');
grid on;



% Make the animation
fig_anim = figure;
ax_anim = axes(fig_anim);
xlim(ax_anim, [min(bins), max(bins)]);
ylim(ax_anim, [0, 1]);
ax_anim.YAxis.Visible = 'off';
ax_anim.XAxis.Visible = 'off';
hold(ax_anim, 'on');

xdata = [];
ydata = [];
ln = scatter(ax_anim, xdata, ydata, 'filled');

for idx = 1:length(bins)
    xdata = [xdata, bins(idx)];
    ydata = [ydata, counter(idx)];
    offset = ydata / max(ydata);
    set(ln, 'XData', xdata, 'YData', offset);
    drawnow;
end



% Define the Mixture of Gaussians PDF
function p = mixture_of_gaussians_pdf(x)
    % Two standard normal distributions, centered at +2 and -2
    components = [normpdf(x, -2, 1), normpdf(x, 2, 1)];
    weights = [0.5, 0.5];
    p = weights * components';
end

% Implement Metropolis sampler
function samples = metropolis_sample(pdf, steps, step_size, init)
    % Metropolis sampler with a normal proposal
    point = init;
    samples = zeros(1, steps);
    for i = 1:steps
        proposed = normrnd(point, step_size);
        if rand() < pdf(proposed) / pdf(point)
            point = proposed;
        end
        samples(i) = point;
    end
end


