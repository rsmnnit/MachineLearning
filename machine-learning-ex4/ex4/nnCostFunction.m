function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


delta2 = zeros(m,size(Theta1,1)+1);
delta3 = zeros(m,size(Theta2,1));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


X = X';
X = [ones(1,size(X,2));X];

Xtheta1 = Theta1 * X;
layer2 = zeros(size(Theta1,1),1);
layer2 = sigmoid(Xtheta1);
layer2 = [ones(1,size(layer2,2));layer2];


layer3 = zeros(size(Theta2,1),1);
Xtheta2 = Theta2*layer2;
layer3 = sigmoid(Xtheta2);

layer3 = layer3';

for i=1:m
  
  for j=1:num_labels
    p = (y(i)==j);
    
    J = J - p*log(layer3(i,j))-(1-p)*log(1-layer3(i,j));
endfor


endfor
J = J/m;

reg = 0;
[u v] = size(Theta1);

for i=1:u
  for j=2:v
    reg =reg + power(Theta1(i,j),2);
endfor
endfor

[u v] = size(Theta2);

for i=1:u
  for j=2:v
    reg =reg + power(Theta2(i,j),2);
endfor
endfor

J = J + reg*lambda/(2.0*m);


#backpropagation

for i=1:m
  p = zeros(size(layer3,2),1);
  p(y(i),1)=1;
delta3(i,:) = layer3(i,:)-p';
endfor

Xtheta1 = Xtheta1';


  
  Xtheta1 = [ones(size(Xtheta1,1),1),Xtheta1];
 
for i=1:m
 
  delta2(i,:) = ((Theta2'*delta3(i,:)').*sigmoidGradient(Xtheta1(i,:))')';
endfor
delta2 = delta2';
delta2(1,:) = [];
#X(1,:)=[];
Theta1_grad = (delta2*X')/m;

delta3 = delta3';
#layer2(1,:) = [];

Theta2_grad = (delta3*layer2')/m;



[u v] = size(Theta1);

for i=1:u
  tmp = 0;
  for j=2:v
    
    tmp =  Theta1(i,j);
    Theta1_grad(i,j) = Theta1_grad(i,j)+ lambda*tmp/m;
  endfor
  
endfor



tmp = 0;
[u v] = size(Theta2);
for i=1:u
  tmp = 0;
  for j=2:v
    
    tmp =  Theta2(i,j);
    Theta2_grad(i,j) = Theta2_grad(i,j) + lambda*tmp/m;
  endfor
  

endfor



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
