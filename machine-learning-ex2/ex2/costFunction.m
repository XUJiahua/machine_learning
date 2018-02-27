function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

y_hat = sigmoid(X * theta);
% (m, 1)
err = y_hat - y;
% (n, 1) = (n, m) * (m, 1)
grad = transpose(X) * err / m;

% for loop version
% for i = 1:m
%     yi_hat = y_hat(i);
%     J = J + (-1 * y(i) * log(yi_hat) - (1 - y(i)) * log(1-yi_hat));
% end
% J = J / m;

% vectorilization version
J = sum(-1 * y .* log(y_hat) - (1 - y) .* log(1-y_hat)) / m;

% =============================================================

end
