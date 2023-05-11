clear; 
%% Given Data
% y_d(m,1) is oil consumption 
y_d = readmatrix("oil_data.txt");
x1 = readmatrix("temp_data.txt");
x2 = readmatrix("insulation_data.txt");
m = length(y_d);
%% Matrices
X = [ones(m,1),x1,x1.^2,x2,x2.^2,x1.*x2];
syms b0 b1 b2 b3 b4 b5
%% Estimating the Parameters
b = (X' * X)\ X' * y_d;
[b0,b1,b2,b3,b4,b5] = deal(b(1),b(2),b(3),b(4),b(5),b(6));
y = b0 + b1*x1 + b2*(x1.^2) + b3*x2 + b4*(x2.^2) + b5*(x1.*x2);
%% Coeff of determinism
unexplained_sum = sum((y_d - y).^2);
total_sum = sum((y_d - mean(y_d)).^2);
Rsqr = (total_sum - unexplained_sum) / total_sum;
%% Prediction
syms y x1 x2
y = b0 + b1*x1 + b2*(x1.^2) + b3*x2 + b4*(x2.^2) + b5*(x1.*x2);
x1 = 10; % temperation is 5 F
x2 = 5;  % insulation is 5 inches
result = eval(y);
%% Print Results of Linear model
fprintf("\n --------- Quadratic Model ---------\n");
fprintf("\n y = %f %f*x1 %f*x1^2 %f*x2  +%f*x2^2 +%f*(x1*x2)\n\n",...
        b0,b1,b2,b3,b4,b5);
fprintf(" b0 = %f\n b1 = %f\n b2 = %f\n b3 = %f\n b4 = %f\n b5 = %f\n",...
        b0,b1,b2,b3,b4,b5);
fprintf("\n r^2 = %f%%\n",Rsqr*100);
fprintf("\n at temp = 10 F, insulation = 5 inches");
fprintf("\n Prediction = %f\n\n",result);
%% Removing unreasonable data
y_d = [y_d(1:5);y_d(7:m)];
x1 = readmatrix("temp_data.txt");
x1 = [x1(1:5);x1(7:m)];
x2 = readmatrix("insulation_data.txt");
x2 = [x2(1:5);x2(7:m)];
m = length(y_d);
%% Recalculations
X = [ones(m,1),x1,x1.^2,x2,x2.^2,x1.*x2];
b = (X' * X)\ X' * y_d;
[b0,b1,b2,b3,b4,b5] = deal(b(1),b(2),b(3),b(4),b(5),b(6));
y = b0 + b1*x1 + b2*(x1.^2) + b3*x2 + b4*(x2.^2) + b5*(x1.*x2);
unexplained_sum = sum((y_d - y).^2);
total_sum = sum((y_d - mean(y_d)).^2);
Rsqr = (total_sum - unexplained_sum) / total_sum;
syms y x1 x2
y = b0 + b1*x1 + b2*(x1.^2) + b3*x2 + b4*(x2.^2) + b5*(x1.*x2);
x1 = 10; % temperation is 5 F
x2 = 5;  % insulation is 5 inches
result = eval(y);
fprintf(" --- Quadratic: after removing outlier ---");
fprintf("\n y = %f %f*x1 %f*x1^2 %f*x2  +%f*x2^2 +%f*(x1*x2)\n\n",...
        b0,b1,b2,b3,b4,b5);
fprintf(" b0 = %f\n b1 = %f\n b2 = %f\n b3 = %f\n b4 = %f\n b5 = %f\n",...
        b0,b1,b2,b3,b4,b5);
fprintf("\n r^2 = %f%%\n",Rsqr*100);
fprintf("\n at temp = 10 F, insulation = 5 inches");
fprintf("\n Prediction = %f\n\n",result);