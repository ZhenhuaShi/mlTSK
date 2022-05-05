function deltaLY = calculateDeltaLY(Y, YPred, Nbs)
isRegression = size(Y, 2) == 1;
if isRegression
    deltaLY = (YPred - Y) / Nbs; % Mean Squared Error
else
    YPred = softmax(YPred')';
    deltaLY = (YPred - Y) / Nbs; % Mean Cross Entropy
end