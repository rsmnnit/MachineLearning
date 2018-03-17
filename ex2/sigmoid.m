function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

sz = size(z,1);
sz2 = size(z,2);
for i=1:sz
  for j=1:sz2
  g(i,j)= 1/(1+power(e,-z(i,j)));
  endfor



% =============================================================

end
