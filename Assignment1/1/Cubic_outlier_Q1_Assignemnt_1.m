x=[0; 1; 2; 3; 4; 5; 6; 7; 8; 9] ;
y=[14000; 13000; 12000; 11000; 1050; 10000; 9500; 9000; 8700; 8000] ;
p = polyfit(x,y,3);
yCalc = polyval(p,x);
scatter(x,y)
hold on
xlabel('The number of years since 1987 x')
ylabel('The numbers of insured persons y')
title('Cubic Regression Relation Between insured persons & no. of years')
grid on
plot(x,yCalc)
legend('Given Data','Cubic fit','Location','best');
Rsq = 1 - sum((y - yCalc).^2)/sum((y - mean(y)).^2);