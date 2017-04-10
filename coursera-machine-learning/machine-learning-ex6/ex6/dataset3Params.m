function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

Test_Params = [0.01 0.03 0.1 0.3 1 3 10 30];
size_params = size(Test_Params, 2);
errors = zeros(size_params*size_params, 1);
c_s_idx = zeros(size_params*size_params, 2);
i=1;

for c=1:size_params
	for s=1:size_params
		model= svmTrain(X, y, Test_Params(c), @(x1, x2) gaussianKernel(x1, x2, Test_Params(s)));
		
		predictions = svmPredict(model, Xval);
		
		errors(i) = mean(double(predictions ~= yval));
		
		c_s_idx(i, 1) = c;
		c_s_idx(i, 2) = s;
		
		i++;
	end
end

index = (min(errors) == errors);

C = Test_Params(c_s_idx(index, 1));
sigma = Test_Params(c_s_idx(index, 2));



% =========================================================================

end
