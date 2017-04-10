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

hypo_term = 0;
lambda_term = 0;

hypo_term_gradient = 0;
lambda_term_gradient= 0;


for i=1:m
	hx = X(i, :)*theta;
	hypo_term += (hx - y(i))^2;
end

n = size(theta, 1);

for j=1:n
	if j >= 2 
		lambda_term += theta(j)^2;
	end

	hypo_term_gradient = 0;

	for i=1:m
		hx = X(i, :)*theta;
		hypo_term_gradient += ((hx - y(i)) * X(i, j));
	end

	if j >= 2 
		lambda_term_gradient = theta(j);
	end

	grad(j) = hypo_term_gradient/m + lambda_term_gradient*(lambda/m);
end

hypo_term /= (2*m);
lambda_term /= (2*m);

J = hypo_term + lambda_term*lambda;

% =========================================================================

grad = grad(:);

end
