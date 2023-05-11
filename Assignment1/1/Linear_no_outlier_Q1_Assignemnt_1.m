x=[0; 1; 2; 3; 5; 6; 7; 8; 9] ;
y=[14000; 13000; 12000; 11000; 10000; 9500; 9000; 8700; 8000] ;
scatter(x,y)
hold on
xlabel('The number of years since 1987 x')
ylabel('The numbers of insured persons y')
title('Linear Regression Relation Between insured persons & no. of years')
grid on
x=[ones(length(x),1) x] ;
B=mldivide(x,y) ;
yCalc = x*B;
x(:,1)=[] ;
plot(x,yCalc)
legend('Given Data','Linear fit','Location','best');
Rsq = 1 - sum((y - yCalc).^2)/sum((y - mean(y)).^2);