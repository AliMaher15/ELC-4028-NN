clear; 
%% Given Data
y_d = readmatrix("height_data.txt");
x = readmatrix("time_data.txt");
x(1) = 0.0001;
m = length(y_d);
%% Matrices
X = [ones(m,1),log(x)];
syms b0 b1
%% Estimating the Parameters
b = (X' * X)\ X' * y_d;
[b0,b1] = deal(b(1),b(2));
%% Scatter Plot
subplot(2,2,4)
scatter(x, y_d, 'LineWidth', 1.5);
hold on;
xlabel('Time (sec)'); ylabel('Height (meter)');
title("Power",'FontSize',16,'FontWeight','bold');
xlim([-0.2 1]);
grid on;
y = b0 + b1*log(x);
plot(x, y, 'LineWidth', 1.5);
hold off;
%% Coeff of determinism
unexplained_sum = sum((y_d - y).^2);
total_sum = sum((y_d - mean(y_d)).^2);
Rsqr = abs(total_sum - unexplained_sum) / total_sum;
%% Print Results of Linear model
fprintf("\n --------- Power Model ---------\n");
fprintf("\n y = %f*x^(%f)\n\n",b0,b1);
fprintf(" b0 = %f\n b1 = %f\n\n",b0,b1);
fprintf("\n r^2 = %f%%\n\n",Rsqr*100);