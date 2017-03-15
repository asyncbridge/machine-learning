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
n = size(X)(1,2);
sum_term = 0;

for i=1:m
	hypo = sigmoid(dot(theta', X(i, :)));

	sum_term = sum_term + ( -y(i)*log(hypo) - (1-y(i))*log(1-hypo) );
end

sum_term_lambda = 0;

% Should not regularize theta(1)
for j=2:n
	sum_term_lambda = sum_term_lambda + theta(j)^2;
end

J = 1/m*sum_term + (lambda*sum_term_lambda)/(2*m);
sum_term_grad = 0;

for j=1:n
	sum_term_grad = 0;
	
	for i=1:m
		hypo = sigmoid(dot(theta', X(i, :)));
		
		sum_term_grad = sum_term_grad + ( hypo - y(i) )* X(i, j);
	end
	
	% Should not regularize theta(1)
	if j == 1
		grad(j) = sum_term_grad/m;
	
	else	  
		grad(j) = sum_term_grad/m + lambda*theta(j)/m;
	
end

% =============================================================

end
