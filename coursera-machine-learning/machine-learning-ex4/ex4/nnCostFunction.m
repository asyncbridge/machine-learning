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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate J function.

% logistic regression term
logstic_term = 0;

% Delta1,2
D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));

% Matrix sizes
% m = 5000 (# of training set)
% hidden_layer_size = 25
% input_layer_size = 400
% num_labels = 10
%
% a1 = 5000x401 (+1 bias unit)
% a2 = 5000x26 (+1 bias unit)
% a3 = 5000x10
% y = 5000x10
%
% Theta1 = 25x401 (+1 bias unit)
% Theta2 = 10x26 (+1 bias unit)

for t=1:m

	% Run feedforward and get hypothesis function.
	a1 = [1 X(t, :)];
	z2 = a1*Theta1';
	a2 = sigmoid(z2);
	a2 = [1 a2];
	hx = a3 = sigmoid(a2*Theta2');

	% Convert ith training set's y to vector(1 by k).
    % Set 1 to y_ in each label using with y(i) value as an index.
	y_ = [ zeros(1, num_labels) ];
	y_(1, y(t)) = 1;
 
	% Calculate logstic term
	for k=1:num_labels
		logstic_term += ( -y_(k)*log(hx(k)) - (1 - y_(k))*log( 1 - hx(k) ) ); 
	end
 
	% Run backpropagation
 
	% Layer 3, the output layer
	delta3 = a3(:) - y_(:);
	
	% Layer 2, the hidden layer
	delta2 = (Theta2(:,2:end)'*delta3)' .* sigmoidGradient(z2);
	
	% Accumulate gradient
	D1 = D1 + delta2'*a1;
	D2 = D2 + delta3*a2;
end

logstic_term /= m;

% logistic regression regularization term
lambda_term = 0;
lambda_term1 = 0;
lambda_term2 = 0;

for j=1:hidden_layer_size
 for k=2:input_layer_size+1
  lambda_term1 += Theta1(j,k)^2;
 end
end 

for j=1:num_labels
 for k=2:hidden_layer_size+1
  lambda_term2 += Theta2(j,k)^2;
 end
end 

lambda_term = (lambda*(lambda_term1+lambda_term2))/(2*m);

J = logstic_term + lambda_term;

% Gradient lambda term
Theta1_ = [ zeros(size(Theta1,1), 1) Theta1(:,2:end)];
Theta2_ = [ zeros(size(Theta2,1), 1) Theta2(:,2:end)];
Theta1_grad_lambda = Theta1_;
Theta2_grad_lambda = Theta2_;

Theta1_grad = D1/m + Theta1_grad_lambda*lambda/m;
Theta2_grad = D2/m + Theta2_grad_lambda*lambda/m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
