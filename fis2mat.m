function [AntecedentsB, WB]=fis2mat(fis)
% from sugfis object to matrix
if nargin < 1
    rng(0)
    warning off all
    load('./fis_results.mat','fis')
end
NumInput=length(fis.Inputs);
NumMF=length(fis.Inputs(1, 1).MembershipFunctions);
NumOutput=length(fis.Outputs);
AntecedentsB=nan(NumMF,NumInput,2);
WB=nan(NumMF,length(fis.Outputs(1).MembershipFunctions(1).Parameters),NumOutput);
for i=1:NumInput
    for j=1:NumMF
        AntecedentsB(j,i,:) = fis.Inputs(i).MembershipFunctions(j).Parameters;
    end
end
for i=1:NumOutput
    for j=1:NumMF
        WB(j,[2:end 1],i) = fis.Outputs(i).MembershipFunctions(j).Parameters;
    end
end
% data=load('./mat_results.mat','AntecedentsB', 'WB');
% sum(abs(data.AntecedentsB-AntecedentsB),'all')
% sum(abs(data.WB-WB),'all')
end