
clear 
close all
clc

% Set seed for reproducibility
rng(1);

% Define the layout for subplots
figure;
subplot(2, 2, 1);
set(gca, 'FontSize', 12);
set(gca, 'LineWidth', 1.5);
set(gca, 'XTick', []);
set(gca, 'YTick', []);
d1 = binornd(4, 0.8, [1, 50]);
t1 = tabulate(d1);
bar(t1(:, 1), t1(:, 2), 'FaceColor', '#3690C0');
ylabel('Counts');
xlabel('Number of Bleaching Steps');
text(1, 18, '\theta = 0.8', 'FontSize', 13);

subplot(2, 2, 2);
set(gca, 'FontSize', 12);
set(gca, 'LineWidth', 1.5);
set(gca, 'XTick', []);
set(gca, 'YTick', []);
A = sum(d1);
B = sum(4 - d1);
theta = linspace(0, 1, 1000);
plot(theta, betapdf(theta, A, B), 'LineWidth', 3, 'Color', '#3690C0');
xlabel('\theta');
ylabel(['p(\theta|', 'y[N]', ')']);
title('\theta = 0.8');

subplot(2, 2, 3);
set(gca, 'FontSize', 12);
set(gca, 'LineWidth', 1.5);
set(gca, 'XTick', []);
set(gca, 'YTick', []);
d2 = binornd(4, 0.5, [1, 50]);
t2 = tabulate(d2);
bar(t2(:, 1), t2(:, 2), 'FaceColor', '#3690C0');
ylabel('Counts');
xlabel('Number of Bleaching Steps');
text(1, 18, '\theta = 0.5', 'FontSize', 13);

subplot(2, 2, 4);
set(gca, 'FontSize', 12);
set(gca, 'LineWidth', 1.5);
set(gca, 'XTick', []);
set(gca, 'YTick', []);
A = sum(d2);
B = sum(4 - d2);
plot(theta, betapdf(theta, A, B), 'LineWidth', 3, 'Color', '#3690C0');
xlabel('\theta');
ylabel(['p(\theta|', 'y[N]', ')']);
title('\theta = 0.5');
