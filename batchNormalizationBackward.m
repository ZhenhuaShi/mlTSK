function [dX,dBeta,dGamma] = batchNormalizationBackward(dZ, X, gamma, ...
    epsilon, batchMean, invSqrtVarPlusEps, channelDim) %#ok<INUSL>
% Back-propagation using batch normalization layer on the host
% NB: invSqrtVarPlusEps is 1./sqrt(var(X) + epsilon)

%   Copyright 2016-2018 The MathWorks, Inc.

% We need to take statistics over all dimensions except the activations 
% (third dimension for 4-D array/fourth dimesnion for 5-D array)
reduceDims = [1:channelDim-1 channelDim+1:ndims(X)];
m = numel(X) ./ size(X, channelDim); % total number of elements in batch per activation

Xnorm = (X - batchMean) .* invSqrtVarPlusEps;

% Get the gradient of the function w.r.t the parameters beta and gamma.
dBeta = sum(dZ, reduceDims);
dGamma = sum(dZ .* Xnorm, reduceDims);

% Now get the gradient of the function w.r.t. input (x)
% See Ioffe & Szegedy, "Batch Normalization: Accelerating Deep Network
% Training by Reducing Internal Covariate Shift" for details.

factor = gamma .* invSqrtVarPlusEps;
factorScaled = factor ./ m;

dMean = dBeta .* factorScaled;
dVar = dGamma .* factorScaled;

dX = dZ .* factor - Xnorm .* dVar - dMean;

end
