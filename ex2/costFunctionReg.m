function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


sz = size(theta);


for j = 1:sz
  tmp2 = 0;
  for i = 1:m
    tmp2 = tmp2 + (1/(1+power(e,-theta'*X(i,:)'))-y(i))*X(i,j);

  endfor
  grad(j) = tmp2/m;
  if(j>1)
  grad(j) = grad(j) + lambda*theta(j)/m;
  endif
  endfor

 for i = 1:m
   J = J -y(i)*log(1/(1+power(e,-theta'*X(i,:)'))) -(1-y(i))*log(1-1/(1+power(e,-theta'*X(i,:)')));
endfor
J=J/m;
tmp = 0;
  for i=2:sz
    tmp = tmp + theta(i)*theta(i);
  endfor
  tmp = tmp * lambda/(2*m);
  J=J+tmp;





% =============================================================

end
