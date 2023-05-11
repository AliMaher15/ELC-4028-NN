clear; 
%% Given Data
y_d = readmatrix("height_data.txt");
x = readmatrix("time_data.txt");
m = length(y_d);
%% Matrices
X = [ones(m,1),x];
syms b0 b1
%% Estimating the Parameters
b = (X' * X)\ X' * y_d;
[b0,b1] = deal(b(1),b(2));
%% Scatter Plot
figure(1)
subplot(2,2,1)
scatter(x, y_d, 'LineWidth', 1.5);
hold on;
xlabel('Time (sec)'); ylabel('Height (meter)');
title("Linear",'FontSize',16,'FontWeight','bold');
xlim([-0.2 1]);
grid on;
y = b0 + b1*x;
plot(x, y, 'LineWidth', 1.5);
hold off;
%% Coeff of determinism
unexplained_sum = sum((y_d - y).^2);
total_sum = sum((y_d - mean(y_d)).^2);
Rsqr = (total_sum - unexplained_sum) / total_sum;
%% Print Results of Linear model
fprintf("\n --------- Linear Model ---------\n");
fprintf("\n y = %f + %f*x\n\n",b0,b1);
fprintf(" b0 = %f\n b1 = %f\n\n",b0,b1);
fprintf("\n r^2 = %f%%\n\n",Rsqr*100);