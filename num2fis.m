function fis=num2fis(AntecedentsB, WB, WAB)
% from matrix to sugfis object
if nargin < 1
    rng(0)
    warning off all
    load('./mat_results.mat','AntecedentsB', 'WB', 'WAB')
end
fis = sugfis;
NumInput=size(AntecedentsB,2);
NumMF=size(AntecedentsB,1);
NumOutput=size(WB,3);
for i=1:NumInput
    for j=1:NumMF
        fis.Inputs(i).MembershipFunctions(j) = fismf("gaussmf", squeeze(AntecedentsB(j,i,:))');
    end
end
for i=1:NumOutput
    for j=1:NumMF
        fis.Outputs(i).MembershipFunctions(j) = fismf("linear", squeeze(WB(j,[2:end 1],i)));
    end
end
rules = fisrule([repmat((1:NumMF)',1,NumInput+NumOutput) ones(NumMF,2)],NumInput);
rules = update(rules,fis);
fis.Rules = rules;
save('./fis_results.mat','fis','WAB')
% plotfis(fis)
% plotmf(fis,'input',1)
% % plotmf(fis,'output',1)
% % fis.Rules
% gensurf(fis)
% load('Estate-costs.mat','yTrain','XTune','yTune','XTest','yTest')
% yTune=(yTune-mean(yTrain))/std(yTrain);
% yTest=(yTest-mean(yTrain))/std(yTrain);
% yTrain=(yTrain-mean(yTrain))/std(yTrain);
% yPredTest=evalfis(fis,[ones(size(XTune,1),1) XTune]*WAB);
% yPredTest(isnan(yPredTest)) = nanmean(yPredTest);
% sqrt((yTune-yPredTest)'*(yTune - yPredTest)/length(yTune))
% yPredTest=evalfis(fis,[ones(size(XTest,1),1) XTest]*WAB);
% yPredTest(isnan(yPredTest)) = nanmean(yPredTest);
% sqrt((yTest-yPredTest)'*(yTest - yPredTest)/length(yTest))
end