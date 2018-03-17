function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


sz  = size(X,2);
szi = size(theta);
for i=1:m
J = J +  sum(power(sum(theta'.*X(i,:))-y(i),2));
for j=1:szi
grad(j) = grad(j) + (sum(sum(theta'.*X(i,:))-y(i)))*X(i,j);
endfor

endfor
tmp = 0;
sz = size(theta);
grad(1) = grad(1)/m;

for i=2:sz
  grad(i) = grad(i)/m;
  tmp =tmp+ power(theta(i),2);
  grad(i) = grad(i) + lambda*theta(i)/m;
  
  
endfor
J = (J+tmp*lambda)/(2*m);



% =========================================================================

grad = grad(:);

end
