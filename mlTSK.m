function varargout = mlTSK(trainX, trainY, testX, testY, varargin)
isDebug = false;
if nargin < 4
    isDebug = true;
    % prepare dataset
    isRegression = datasample(0:1, 1);
    if isRegression
        load Musk1
    else
        load Musk1C
    end
    [N, M] = size(trainX); % N samples, M features
    tuneN = size(tuneY, 1);
    testN = size(testY, 1);
    nC = size(trainY, 2);
    testX = {tuneX, testX};
    testY = {tuneY, testY};
    fprintf("train on %d samples, tune on %d samples, test on %d samples, num. of features is %d.\n", ...
        N, tuneN, testN, M)
    if isRegression
        fprintf("Regression Task.\n")
    else
        fprintf("Classification Task, num. of class is %d.\n", nC)
    end
    % control random number generator for reproducibility
    rng(0)
end
nC = size(trainY, 2);
isRegression = nC == 1; % Regression or Classification
thre0 = (isRegression - 0.5) * inf; % threshold for parameter validation
[N, M] = size(trainX); % N samples, M features
% running moving average estimate for BatchNorm if N is large enough
RunningEstimate = N > 1e4;
% special case: validation set unavailable
if ~iscell(testX)
    testX = {testX};
    testY = {testY};
end
if isRegression
    % normalize the regression labels
    mY = mean(trainY);
    sY = std(trainY);
    trainY = (trainY - mY) / sY;
    testY = cellfun(@(u)(u - mY)/sY, testY, 'UniformOutput', false);
end
% variable input
if nargin > 4
    [varargin{:}] = convertStringsToChars(varargin{:});
end
% Parse arguments and check if parameter/value pairs are valid
paramNames = {'MF', 'DR', 'nF', 'Init', 'nMF', 'Uncertain', 'TR', ...
    'Opt', 'Powerball', 'DropRule', 'lr', 'l2', 'nIt', 'Nbs', 'UR', 'RP'};
% CDR-FCM-RDpA, by default
defaults   = {'Gaussian', 'CDR', 4, 'FCM', 16, 'None', 'km', 'AdaBelief', 0.5, 0.5, 0.01, 1e-4, 1000, 64, 0, nan};
% % MBGD-RDA, <Optimize TSK fuzzy systems for regression problems: Mini-batch gradient descent with 
% % regularization, DropRule, and AdaBound (MBGD-RDA)>
% defaults   = {'Gaussian', 'None', 4, 'GP', 4, 'None', 'km', 'AdaBound', 0.5, 0.5, 0.01, 1e-4, 1000, 64, 0, nan};
% % FCM-RDpA, <FCM-RDpA: TSK fuzzy regression model construction using fuzzy c-means clustering, 
% % regularization, DropRule, and Powerball AdaBelief>
% defaults   = {'Gaussian', 'None', 4, 'FCM', 16, 'None', 'km', 'AdaBelief', 0.5, 0.5, 0.01, 1e-4, 1000, 64, 0, nan};
% % FCM-RDpA-UR-BNC, <Optimize TSK fuzzy systems for classification problems: Mini-batch gradient 
% % descent with uniform regularization and batch normalization>
% defaults   = {'Gaussian', 'BNC', 4, 'FCM', 16, 'None', 'km', 'AdaBelief', 0.5, 0.5, 0.01, 0.05, 1000, 64, 1, nan};
% % FCM-RDpA-LN-ReLU
% defaults   = {'Gaussian', 'None', 4, 'FCM-LN-ReLU', 16, 'None', 'km', 'AdaBelief', 0.5, 0.5, 0.01, 0.05, 1000, 64, 1, nan};
% % CDR-FCM-RDpA, by default
% defaults   = {'Gaussian', 'CDR', 4, 'FCM', 16, 'None', 'km', 'AdaBelief', 0.5, 0.5, 0.01, 1e-4, 1000, 64, 0, nan};
% % CDRP-FCM-RDpA
% defaults   = {'Gaussian', 'CDR', 4, 'FCM', 16, 'None', 'km', 'AdaBelief', 0.5, 0.5, 0.01, 1e-4, 1000, 64, 0, [0, 500]};
% % CFS-FCM-RDpA
% defaults   = {'Gaussian', 'CFS', 4, 'FCM', 16, 'None', 'km', 'AdaBelief', 0.5, 0.5, 0.01, 1e-4, 1000, 64, 0, nan};
% % CFSP-FCM-RDpA
% defaults   = {'Gaussian', 'CFS', 4, 'FCM', 16, 'None', 'km', 'AdaBelief', 0.5, 0.5, 0.01, 1e-4, 1000, 64, 0, [0, 500]};


[MF, DR, nF, Init, nMF, Uncertain, TR, Opt, Powerball, DropRule, lr, l2, nIt, Nbs, UR, RP] ...
    = internal.stats.parseArgs(paramNames, defaults, varargin{:});
% mini-batch size
if Nbs > N
    Nbs = N;
end
% subspace dimensionality
if nF > M
    nF = M;
end

%% Structure Selection: shared or independent MembershipFunctions (MFs)
% identify the number of rules (nRule) for different structure, set nMF as the structure flag
if ismember(Init, {'GP', 'Grid Partition'}) % isShare==nMF
    % for shared MFs, each feature has 2 MFs by default
    if nMF == 0
        nMF = 2;
    end
    nRule = nMF^nF;
else
    % for independent MFs, nRule equals the number of MFs on each feature
    nRule = nMF;
    nMF = 0;
end
% initialize AdaptiveRulePruning (ARP) setting, pruning half rules by default
% RP stores the ARP flag, and the number of iterations to perform pruning
isARP = 0;
if RP(1) == 0
    isARP = 1;
    RP = RP(2:end);
end
nRuleStep = nRule / 2 / length(RP); % number of pruned rules for each pruning
% set the L_{2,1} (GroupLasso) regularized coefficient the same as L2 coefficient
FS = l2;

% Check settings
if nMF && ismember(DR, {'FA1', 'FA2', 'FAx'})
    error('FeatureAugmentation conflicts with shared MFs.')
end
if ismember(Init, {'FCM-LN', 'FCM-LN-ReLU'}) && DropRule > 0
    warning('Removing DropRule for LayerNorm...')
    DropRule = 0;
end
if nIt < 20
    error('Too few iterations to converge.')
end
if nIt > 20000
    error('Too many iterations to complete training in a reasonable time.')
end
nPoints = fix(nIt/20+1);
iterationsRecorded = fix(linspace(1, nIt, nPoints));
if nRule > 256
    error('Too many rules to complete training in a reasonable time.')
end

%% Parameter Initialization
% DimensionalityReduction Layer Initialization
switch DR
    case {'None'}
        % None: without DR
        trainXA = trainX;
        [WA0, WC0] = deal([zeros(1, M); eye(M)]);
        [MA, MC] = deal(M);
        nF = M;
    case {'CBN', 'BNA', 'BNC'}
        % CBN: Consistant BatchNorm
        % BNA: BatchNorm for Antecedents
        % BNC: BatchNorm for Consequents
        % running moving average estimate for BatchNorm if N is large enough
        if RunningEstimate
            trainXA = trainX;
        else
            [trainXA, trainXmean, trainXstd] = zscore(trainX);
        end
        [WA0, WC0] = deal([zeros(1, M); eye(M)]);
        [MA, MC] = deal(M);
        nF = M;
    case {'DR'}
        % DR: DimensionalityReduction
        [WPCA, trainXA] = pca(trainX, 'NumComponents', nF);
        WA0 = [zeros(1, nF); WPCA];
        WC0 = [zeros(1, M); eye(M)];
        [MA, MC] = deal(nF, M);
    case {'PCAfixed', 'CDR', 'FS', 'CFS'} % CDR==PCAinit
        % PCAfixed: initialized by PCA, untrainable
        % CDR: ConsistentDR, initialized by PCA, trainable
        % FS: FeatureSelection
        % CFS: ConsistentFS
        [WPCA, trainXA] = pca(trainX, 'NumComponents', nF);
        [WA0, WC0] = deal([zeros(1, nF); WPCA]);
        [MA, MC] = deal(nF);
    case {'FA1'}
        % FA1: FeatureAugmentation for Antecedents
        [WPCA, trainXA] = pca(trainX, 'NumComponents', nF);
        trainXA = [trainXA, trainX];
        WA0 = [zeros(1, nF); WPCA];
        WC0 = [zeros(1, M); eye(M)];
        [MA, MC] = deal(nF+M, M);
    case {'FA2', 'FAx'}
        % FA2: Same FeatureAugmentation for Antecedents and Consequents
        % FAx: Different FeatureAugmentation for Antecedents and Consequents
        [WPCA, trainXA] = pca(trainX, 'NumComponents', nF);
        trainXA = [trainXA, trainX];
        [WA0, WC0] = deal([zeros(1, nF); WPCA]);
        [MA, MC] = deal(nF+M);
    case {'RandFixed', 'RandInit'}
        % RandFixed: initialized by orth-rand, untrainable
        % RandInit: initialized by orth-rand, trainable
        WPCA = orth(rand(M, nF));
        trainXA = trainX * WPCA;
        [WA0, WC0] = deal([zeros(1, nF); WPCA]);
        [MA, MC] = deal(nF);
    case {'PCAxyFixed', 'PCAxyInit'}
        % PCAxyFixed: initialized by PCAxy, untrainable
        % PCAxyInit: initialized by PCAxy, trainable
        WPCA = pca([trainX, trainY], 'NumComponents', nF);
        WPCA = WPCA(1:end-nC, :);
        trainXA = trainX * WPCA;
        [WA0, WC0] = deal([zeros(1, nF); WPCA]);
        [MA, MC] = deal(nF);
end

% Antecedent and Consequent Initialization
switch Init
    case {'Rand'} % Random Initialization
        C0 = 2 * rand(nRule, MA) - 1;
        Sigma0 = 5 * rand(nRule, MA);
        W0 = 2 * rand(nRule, MC+1, nC) - 1; % Rule consequents
    case {'Grid Partition', 'GP'} % GridPartition Initialization
        C0 = zeros(nF, nMF);
        Sigma0 = C0;
        W0 = zeros(nRule, MC+1, nC); % Rule consequents
        for m = 1:nF
            C0(m, :) = linspace(min(trainXA(:, m)), max(trainXA(:, m)), nMF);
            Sigma0(m, :) = std(trainXA(:, m));
        end
    case {'FCM', 'FCMx', 'LogFCM', 'HFCM', 'LogHFCM', 'FCM-LN', 'FCM-LN-ReLU'}
        % FCM==FCMx: FuzzyCMeans on the data
        % HFCM: enlarge variance for high-dimensional input initialization,
        % following <Curse of dimensionality for TSK fuzzy neural networks: Explanation and solutions>
        W0 = zeros(nRule, MC+1, nC); % Rule consequents
        % FCM initialization
        [C0, U] = FuzzyCMeans(trainXA, nRule, [2, 100, 0.001, 0]);
        Sigma0 = C0;
        for ir = 1:nRule
            Sigma0(ir, :) = std(trainXA, U(ir, :));
            W0(ir, 1, :) = U(ir, :) * trainY / sum(U(ir, :));
        end
        if ismember(Init, {'HFCM', 'LogHFCM'})
            % HFCM: enlarge variance
            Sigma0 = sqrt(size(trainXA, 2)) * Sigma0;
        end
    case {'FCMy'} % FuzzyCMeans on the labels
        W0 = zeros(nRule, MC+1, nC); % Rule consequents
        % FCM initialization
        [W0(:, 1, :), U] = FuzzyCMeans(trainY, nRule, [2, 100, 0.001, 0]);
        C0 = U * trainXA;
        Sigma0 = C0;
        for ir = 1:nRule
            Sigma0(ir, :) = std(trainXA, U(ir, :));
        end
    case {'kMx'} % kMeans on the data
        W0 = zeros(nRule, MC+1, nC); % Rule consequents
        [ids, C0] = kmeans(trainXA, nRule, 'replicate', 3);
        Sigma0 = C0;
        for ir = 1:nRule
            Sigma0(ir, :) = std(trainXA(ids == ir, :));
            W0(ir, 1, :) = mean(trainY(ids == ir, :));
        end
    case {'kMy'} % kMeans on the labels
        W0 = zeros(nRule, MC+1, nC); % Rule consequents
        [ids, W0(:, 1, :)] = kmeans(trainY, nRule, 'replicate', 3);
        [C0, Sigma0] = deal(zeros(nRule, MA));
        for ir = 1:nRule
            C0(ir, :) = mean(trainXA(ids == ir, :));
            Sigma0(ir, :) = std(trainXA(ids == ir, :));
        end
end
Sigma0(Sigma0 == 0) = mean(Sigma0(:)); % avoid zero variance

% Antecedent Initialization for different MembershipFunctions
switch Uncertain
    case {'None'} % type-1 fuzzy set
        switch MF % MembershipFunctions
            case {'Bell-Shaped', 'gbell'}
                Antecedents0 = cat(3, Sigma0, 5*ones(size(C0)), C0);
            case {'Gaussian', 'gauss'}
                Antecedents0 = cat(3, Sigma0, C0);
            case {'Trapezoidal', 'trap'}
                [A0, D0] = deal(C0-10*Sigma0, C0+10*Sigma0);
                [B0, C0] = deal(C0-.5*Sigma0, C0+.5*Sigma0);
                Antecedents0 = cat(3, A0, B0, C0, D0);
            case {'Triangular', 'tri'}
                [A0, D0] = deal(C0-10*Sigma0, C0+10*Sigma0);
                Antecedents0 = cat(3, A0, C0, D0);
        end
    case {'Mean', 'mean'} % interval type-2 fuzzy set with uncertain mean
        switch MF
            case {'Gaussian', 'gauss'}
                Antecedents0 = cat(3, Sigma0, C0-0.01, C0+0.01);
        end
    case {'Variance', 'var'} % interval type-2 fuzzy set with uncertain variance
        switch MF
            case {'Gaussian', 'gauss'}
                Antecedents0 = cat(3, Sigma0-0.01, Sigma0+0.01, C0);
        end
end
[Antecedents, W, WA, WC] = deal(Antecedents0, W0, WA0, WC0);
[AntecedentsB, WB, WAB, WCB] = deal(Antecedents0, W0, WA0, WC0);
fB = zeros(1, nRule);

[b1, b2, sp, momentum] = deal(0.9, 0.999, 10^(-8), 0.1);
[gammaA, betaA, gammaC, betaC, gammaL, betaL] = deal(1, 0, 1, 0, 1, 0);
trainResult = zeros(1, nPoints);
testResult = cellfun(@(u)trainResult, testY, 'UniformOutput', false);
% number of updating/recording iteration, threshold for parameter validation
[uit, rit, thre] = deal(0, 0, thre0);

%% Parameter Optimization
for it = 1:nIt
    % initialization or re-initialization at the begining, after rule pruning, or for CFS
    if it == 1 || ismember(it, RP) || (ismember(DR, {'CFS'}) && it == fix(nIt / 2))
        switch DR % DimensionalityReductionLayer
            case {'None', 'PCAfixed', 'RandFixed', 'PCAxyFixed'}
                [mea, var] = deal({0, 0}); % two trainable variables: [Antecedents, W]
            case {'CDR', 'FS', 'CFS', 'RandInit', 'PCAxyInit', 'DR', 'FA1', 'FA2'}
                [mea, var] = deal({0, 0, 0}); % three trainable variables: [Antecedents, W, WA]
            case {'FAx', 'CBN', 'BNA', 'BNC'}
                [mea, var] = deal({0, 0, 0, 0}); % four trainable variables: [Antecedents, W, WA/gammaA, WC/betaA]
        end
        if ismember(Init, {'FCM-LN', 'FCM-LN-ReLU'})
            [mea, var] = deal([mea, {0, 0}]); % + two trainable variables: [gammaL, betaL]
        end
        % ConsistentFeatureSelection: GroupLasso+AdaptiveGroupLasso,
        % following <Consistent feature selection for analytic deep neural networks>
        if ismember(DR, {'CFS'}) && it == fix(nIt/2)
            % Adaptive coefficient for 'Consistent' FeatureSelection
            tm = vecnorm(WAB(2:end, :), 2, 2);
            tm = squeeze(1./tm.^2);
            FS = tm * FS;
            % re-initialization
            [Antecedents, W] = deal(AntecedentsB, WB);
            [WA, WC, gammaL, betaL] = deal(WAB, WCB, gammaLB, betaLB);
            [uit, thre] = deal(0, thre0);
        end
        % RulePruning
        if ismember(it, RP)
            if isARP % AdaptiveRulePruning
                % ARP: keep constant the expected number of updated rules [nRule*(1-DropRule)]
                DropRule = 1 - nRule * (1 - DropRule) / (nRule - nRuleStep);
            end
            nRule = nRule - nRuleStep;
            % preserve rules with top-k highest validation FiringLevel
            [~, idx] = maxk(fB, nRule);
            [Antecedents, W] = deal(AntecedentsB(idx, :, :, :), WB(idx, :, :));
            [WA, WC, gammaL, betaL] = deal(WAB, WCB, gammaLB, betaLB);
            [uit, thre] = deal(0, thre0);
        end
    end
    uit = uit + 1; % number of updating iteration
    deltaA = zeros(size(Antecedents));
    deltaW = l2 * [zeros(nRule, 1, nC), W(:, 2:end, :)]; % L2 regularized consequents
    [deltaXA, deltaXC] = deal(zeros(Nbs, nF));
    idsTrain = datasample(1:N, Nbs, 'replace', false); % mini-batch selection
    trainXbatch = trainX(idsTrain, :);
    yReal = trainY(idsTrain, :);
    
    switch DR % DimensionalityReductionLayer Forward
        case {'None'}
            [trainXA, trainXC] = deal(trainXbatch);
        case {'PCAfixed', 'RandFixed', 'PCAxyFixed', 'CDR', 'FS', 'CFS', 'RandInit', 'PCAxyInit'}
            [trainXA, trainXC] = deal([ones(Nbs, 1), trainXbatch]*WA);
        case {'DR'}
            trainXA = [ones(Nbs, 1), trainXbatch] * WA;
            trainXC = trainXbatch;
        case {'CBN', 'BNA', 'BNC'}
            [trainXbn, trainXm, trainXs] = zscore(trainXbatch, 1);
            % running moving average estimate for BatchNorm if N is large enough
            if RunningEstimate
                if it == 1
                    trainXmean = trainXm;
                    trainXstd = trainXs;
                else
                    trainXmean = (1 - momentum) * trainXmean + momentum * trainXm;
                    trainXstd = (1 - momentum) * trainXstd + momentum * trainXs;
                end
            end
            if ismember(DR, {'CBN'}) % CBN: Consistant BatchNorm
                [trainXA, trainXC] = deal([ones(Nbs, 1), trainXbn]*WA);
            elseif ismember(DR, {'BNA'}) % BNA: BatchNorm for Antecedents
                trainXA = [ones(Nbs, 1), trainXbn] * WA;
                trainXC = trainXbatch;
            elseif ismember(DR, {'BNC'}) % BNC: BatchNorm for Consequents
                trainXA = trainXbatch;
                trainXC = [ones(Nbs, 1), trainXbn] * WC;
            end
        case {'FA1'} % FA1: FeatureAugmentation for Antecedents
            trainXA = [[ones(Nbs, 1), trainXbatch] * WA, trainXbatch];
            trainXC = trainXbatch;
        case {'FA2'} % FA2: Same FeatureAugmentation for Antecedents and Consequents
            [trainXA, trainXC] = deal([[ones(Nbs, 1), trainXbatch] * WA, trainXbatch]);
        case {'FAx'} % FAx: Different FeatureAugmentation for Antecedents and Consequents
            trainXA = [[ones(Nbs, 1), trainXbatch] * WA, trainXbatch];
            trainXC = [[ones(Nbs, 1), trainXbatch] * WC, trainXbatch];
    end
    
    yPred = nan(Nbs, nC);
    % For each sample, calculate MembershipFunctions, FiringLevels, Predictions, and Derivatives
    for n = 1:Nbs
        % DropRule: probability of an element to be zeroed, similar with Dropout,
        % following <Optimize TSK fuzzy systems for regression problems:
        % Mini-batch derivative descent with regularization, DropRule, and AdaBound (MBGD-RDA)>
        idsKeep = rand(1, nRule) > DropRule;
        % Calculate FiringLevels using MembershipFunctions
        [fKeep, deltamu] = calculateFiringLevel(trainXA(n, :), Antecedents, idsKeep, MF, nMF, Uncertain, Init);
        % special case: fKeep=inf/-inf/nan; continue with another sample
        if sum(~isfinite(fKeep(:)))
            continue;
        end
        % special case: all fKeep=0; DropRule=0
        if ~sum(fKeep(:))
            idsKeep = true(1, nRule);
            % Calculate FiringLevels using MembershipFunctions
            [fKeep, deltamu] = calculateFiringLevel(trainXA(n, :), Antecedents, idsKeep, MF, nMF, Uncertain, Init);
        end
        yR = permute(sum([1, trainXC(n, :)] .* W(idsKeep, :, :), 2), [3, 1, 2]); % calculate Predictions on each rule
        switch Uncertain
            case {'None'} % type-1 fuzzy set
                if ismember(Init, {'LogFCM', 'LogHFCM'})
                    % LogFCM: logarithm transformed FiringLevel for high-dimensional input,
                    % following <A TSK-type convolutional recurrent fuzzy network for predicting driving fatigue>
                    fKeep = -1 ./ log(fKeep);
                    fBar = fKeep / sum(fKeep);
                    yPred(n, :) = yR * fBar; % averaged prediction
                    deltaLY = calculateDeltaLY(yReal(n, :), yPred(n, :), Nbs); % derivative w.r.t. Y
                    deltaLfKeep = deltaLY * (yR - yR * fBar) / sum(fKeep) .* (fKeep.^2)'; % derivative w.r.t. fKeep
                elseif ismember(Init, {'FCM-LN', 'FCM-LN-ReLU'})
                    % LN: LayerNorm, a variant of BatchNorm,
                    % following <Layer normalization for TSK fuzzy system optimization in regression problems>
                    fBar = fKeep / sum(fKeep);
                    fBarLen = size(fBar, 1);
                    fBarM = mean(fBar);
                    fBarS = std(fBar, 1) + eps;
                    zfBar = (fBar - fBarM) / fBarS;
                    fBarLN = gammaL * zfBar + betaL;
                    if ismember(Init, {'FCM-LN-ReLU'})
                        % ReLU acitivation
                        idsF = fBarLN > 0;
                    else
                        idsF = true(size(fBarLN));
                    end
                    yPred(n, :) = yR(:, idsF) * fBarLN(idsF); % averaged prediction
                    deltaLY = calculateDeltaLY(yReal(n, :), yPred(n, :), Nbs); % derivative w.r.t. Y
                    deltaLfBarLN = zeros(fBarLen, 1);
                    deltaLfBarLN(idsF) = deltaLY * yR(:, idsF); % derivative w.r.t. fBarLN
                    [deltaLfBar, deltaLbetaL, deltaLgammaL] ...
                        = batchNormalizationBackward(deltaLfBarLN, fBar, gammaL, 0, fBarM, 1./fBarS, 2);
                    deltaLfKeep = (deltaLfBar' - deltaLfBar' * fBar) .* fBar'; % derivative w.r.t. fKeep
                else
                    fBar = fKeep / sum(fKeep);
                    yPred(n, :) = yR * fBar; % averaged prediction
                    deltaLY = calculateDeltaLY(yReal(n, :), yPred(n, :), Nbs); % derivative w.r.t. Y
                    deltaLfKeep = deltaLY * (yR - yR * fBar) .* fBar'; % derivative w.r.t. fKeep
                end
                % UR: UniformRegularization, minimize variance of normalized firing levels,
                % following <Optimize TSK fuzzy systems for classification problems:
                % Minibatch derivative descent with uniform regularization and batch normalization>
                if UR
                    nRulesKeep = sum(idsKeep);
                    temp1 = fBar' - 1 / nRulesKeep;
                    temp2 = (temp1 - temp1 * fBar) .* fBar';
                    deltaLfKeep = deltaLfKeep + UR * temp2;
                end
            case {'Mean', 'mean', 'Variance', 'var'} % interval type-2 fuzzy set
                switch TR % TypeReduction
                    case {'Karnik-Mendel', 'km'}
                        % <Centroid of a type-2 fuzzy set>
                        fBar = zeros(size(fKeep, 1), 1);
                        deltaF = zeros([nC, size(fKeep)]);
                        for c = 1:nC
                            [syR, iyR] = sort(yR(c, :));
                            sfl = fKeep(iyR, 1);
                            sfr = fKeep(iyR, 2);
                            % EIASC: Enhanced Iterative Algorithm with Stopping Condition,
                            % <Comparison and practical implementation of type-reduction algorithms
                            % for type-2 fuzzy sets and systems>
                            % <A comprehensive study of the efficiency of type-reduction algorithms>
                            [yPred(n, c), ylPred, yrPred, il, ir] = EIASC(syR, syR, sfl', sfr', 0);
                            fBar(iyR) = fBar(iyR) + ([sfr(1:il); sfl(il + 1:end)] / (sum(sfr(1:il)) + sum(sfl(il + 1:end))) ...
                                +[sfl(1:ir); sfr(ir + 1:end)] / (sum(sfl(1:ir)) + sum(sfr(ir + 1:end)))) / 2;
                            % derivative of Y w.r.t. f
                            % <Computing derivatives in interval type-2 fuzzy logic systems> (17)-(20)
                            deltaf = zeros(size(fKeep));
                            deltaf(iyR(1:il), 2) = (syR(1:il)' - ylPred) / (sum(sfr(1:il)) + sum(sfl(il + 1:end)));
                            deltaf(iyR(il + 1:end), 1) = (syR(il + 1:end)' - ylPred) / (sum(sfr(1:il)) + sum(sfl(il + 1:end)));
                            deltaf(iyR(1:ir), 1) = deltaf(iyR(1:ir), 1) ...
                                +(syR(1:ir)' - yrPred) / (sum(sfl(1:ir)) + sum(sfr(ir + 1:end)));
                            deltaf(iyR(ir + 1:end), 2) = deltaf(iyR(ir + 1:end), 2) ...
                                +(syR(ir + 1:end)' - yrPred) / (sum(sfl(1:ir)) + sum(sfr(ir + 1:end)));
                            deltaF(c, :, :) = deltaf;
                        end
                        deltaLY = calculateDeltaLY(yReal(n, :), yPred(n, :), Nbs); % derivative w.r.t. Y
                        % derivative w.r.t. fKeep
                        deltaLfKeep = 0;
                        for c = 1:nC
                            deltaLfKeep = deltaLfKeep + deltaLY(c) * permute(deltaF(c, :, :), [3, 2, 1]) .* fKeep' / 2;
                        end
                    case {'Nie-Tan'}
                        % <Towards an efficient typereduction method for interval type-2 fuzzy logic systems>
                        fAvg = (fKeep(:, 1) + fKeep(:, 2)) / 2;
                        fBar = fAvg / sum(fAvg);
                        yPred(n, :) = yR * fBar;
                        deltaLY = calculateDeltaLY(yReal(n, :), yPred(n, :), Nbs); % derivative w.r.t. Y
                        deltaLfKeep = deltaLY * (yR - yR * fBar) / sum(fAvg) .* fKeep' / 2; % derivative w.r.t. fKeep
                end
        end
        % special case: deltaLfKeep=inf/-inf/nan; continue with another sample
        if sum(~isfinite(deltaLfKeep(:)))
            continue;
        end
        for c = 1:nC
            if deltaLY(c) ~= 0
                % derivative w.r.t. W
                deltaW(idsKeep, :, c) = deltaW(idsKeep, :, c) + deltaLY(c) * fBar * [1, trainXC(n, :)];
                if ismember(DR, {'CDR', 'CBN', 'BNC', 'FS', 'CFS', 'RandInit', 'PCAxyInit', 'FA2', 'FAx'})
                    % derivative w.r.t. XConsequent
                    deltaXC(n, :) = deltaLY(c) * fBar' * W(idsKeep, 2:1+nF, c);
                end
            end
        end
        switch Uncertain
            case {'None'} % type-1 fuzzy set
                if ~nMF % independent MFs
                    % derivative w.r.t. Antecedent
                    deltaA(idsKeep, :, :) = deltaA(idsKeep, :, :) + deltaLfKeep' .* deltamu;
                    if ismember(DR, {'CDR', 'CBN', 'BNA', 'FS', 'CFS', 'RandInit', 'PCAxyInit', 'DR', 'FA1', 'FA2', 'FAx'})
                        % derivative w.r.t. XAntecedent
                        switch MF % MembershipFuncitons
                            case {'Bell-Shaped', 'gbell', 'Gaussian', 'gauss'}
                                deltaXA(n, :) = -sum(deltaLfKeep'.*deltamu(:, 1:nF, end), 1);
                            case {'Trapezoidal', 'trap'}
                                deltaXA(n, :) = -sum(deltaLfKeep'.*(deltamu(:, 1:nF, 2) + deltamu(:, 1:nF, 3)), 1);
                            case {'Triangular', 'tri'}
                                deltaXA(n, :) = -sum(deltaLfKeep'.*deltamu(:, 1:nF, 2), 1);
                        end
                    end
                else % shared MFs
                    % derivative w.r.t. Antecedent
                    deltaA = deltaA + permute(sum(deltaLfKeep .* deltamu, 2), [1, 3, 4, 2]);
                    if ismember(DR, {'CDR', 'CBN', 'BNA', 'FS', 'CFS', 'RandInit', 'PCAxyInit', 'DR'})
                        % derivative w.r.t. XAntecedent
                        switch MF % MembershipFuncitons
                            case {'Bell-Shaped', 'gbell', 'Gaussian', 'gauss'}
                                deltaXA(n, :) = -sum(permute(sum(deltaLfKeep .* deltamu(:, :, :, end), 2), [1, 3, 4, 2]), 2);
                            case {'Trapezoidal', 'trap'}
                                tmp = sum(deltamu(:, :, :, [2, 3]), 4);
                                deltaXA(n, :) = -sum(permute(sum(deltaLfKeep .* tmp, 2), [1, 3, 4, 2]), 2);
                            case {'Triangular', 'tri'}
                                deltaXA(n, :) = -sum(permute(sum(deltaLfKeep .* deltamu(:, :, :, 2), 2), [1, 3, 4, 2]), 2);
                        end
                    end
                end
            case {'Mean', 'mean', 'Variance', 'var'} % interval type-2 fuzzy set
                % derivative w.r.t. Antecedent
                if ~nMF % independent MFs
                    tmp = sum(deltaLfKeep'.*deltamu, 2);
                    deltaA(idsKeep, :, :) = deltaA(idsKeep, :, :) + permute(tmp, [1, 3, 4, 2]);
                else % shared MFs
                    tmp = sum(permute(deltaLfKeep, [3, 2, 1]).*deltamu, 2);
                    deltaA = deltaA + permute(sum(tmp, 3), [1, 4, 5, 2, 3]);
                end
        end
    end
    % package trainable parameters (param) and derivatives (delta)
    switch DR % DimensionalityReduction
        case {'None', 'PCAfixed', 'RandFixed', 'PCAxyFixed'}
            param = {Antecedents, W};
            delta = {deltaA, deltaW};
        case {'DR', 'FA1'}
            % DR: DimensionalityReduction
            % FA1: FeatureAugmentation for Antecedents
            deltaWA = [ones(Nbs, 1), trainXbatch]' * deltaXA;
            if ismember(DR, {'DR'}) % L2 regularized DR layer
                deltaWA = deltaWA + l2 * [zeros(1, size(WA, 2)); WA(2:end, :)];
            end
            param = {Antecedents, W, WA};
            delta = {deltaA, deltaW, deltaWA};
        case {'CDR', 'FS', 'CFS', 'RandInit', 'PCAxyInit', 'FA2'}
            % FA2: Same FeatureAugmentation for Antecedents and Consequents
            deltaWA = [ones(Nbs, 1), trainXbatch]' * deltaXA;
            deltaWC = [ones(Nbs, 1), trainXbatch]' * deltaXC;
            deltaWA = deltaWA + deltaWC;
            if ismember(DR, {'CDR', 'RandInit', 'PCAxyInit'}) % L2 regularized DR layer
                deltaWA = deltaWA + l2 * [zeros(1, size(WA, 2)); WA(2:end, :)];
            end
            param = {Antecedents, W, WA};
            delta = {deltaA, deltaW, deltaWA};
        case {'CBN'} % CBN: Consistant BatchNorm
            deltabA = sum(deltaXA(:));
            deltagA = deltaXA .* trainXbatch;
            deltagA = sum(deltagA(:));
            deltabC = sum(deltaXC(:));
            deltagC = deltaXC .* trainXbatch;
            deltagC = sum(deltagC(:));
            deltagA = deltagA + deltagC;
            deltabA = deltabA + deltabC;
            param = {Antecedents, W, gammaA, betaA};
            delta = {deltaA, deltaW, deltagA, deltabA};
        case {'BNA'} % BNA: BatchNorm for Antecedents
            deltabA = sum(deltaXA(:));
            deltagA = deltaXA .* trainXbatch;
            deltagA = sum(deltagA(:));
            param = {Antecedents, W, gammaA, betaA};
            delta = {deltaA, deltaW, deltagA, deltabA};
        case {'BNC'} % BNC: BatchNorm for Consequents
            deltabC = sum(deltaXC(:));
            deltagC = deltaXC .* trainXbatch;
            deltagC = sum(deltagC(:));
            param = {Antecedents, W, gammaC, betaC};
            delta = {deltaA, deltaW, deltagC, deltabC};
        case {'FAx'}
            % FAx: Different FeatureAugmentation for Antecedents and Consequents
            deltaWA = [ones(Nbs, 1), trainXbatch]' * deltaXA;
            deltaWC = [ones(Nbs, 1), trainXbatch]' * deltaXC;
            param = {Antecedents, W, WA, WC};
            delta = {deltaA, deltaW, deltaWA, deltaWC};
    end
    if ismember(Init, {'FCM-LN', 'FCM-LN-ReLU'}) % LN: LayerNorm
        [param, delta] = deal([param, {gammaL, betaL}], [delta, {deltaLgammaL, deltaLbetaL}]);
    end
    
    % Powerball \in [0, 1), alleviate the problems of gradient vanishing and gradient explosion,
    % following <On the Powerball method: Variants of descent methods for accelerated optimization>
    delta = cellfun(@(deltaA)sign(deltaA).*(abs(deltaA).^Powerball), delta, 'UniformOutput', false);
    switch Opt % mini-batch gridient descent optimization approach
        case {'AdaBound'}
            % <Adaptive gradient methods with dynamic bound of learning rate>
            lb = lr * (1 - 1 / ((1 - b2) * uit + 1));
            ub = lr * (1 + 1 / ((1 - b2) * uit));
            mea = cellfun(@(deltaA, mA)b1*mA+(1 - b1)*deltaA, delta, mea, 'UniformOutput', false);
            var = cellfun(@(deltaA, vA)b2*vA+(1 - b2)*deltaA.^2, delta, var, 'UniformOutput', false);
            mH = cellfun(@(mA)mA/(1 - b1^uit), mea, 'UniformOutput', false);
            vH = cellfun(@(vA)vA/(1 - b2^uit), var, 'UniformOutput', false);
            lrb = cellfun(@(vAH)min(ub, max(lb, lr ./ (sqrt(vAH) + sp))), vH, 'UniformOutput', false);
            param = cellfun(@(A, lrA, mAH)A-lrA.*mAH, param, lrb, mH, 'UniformOutput', false);
        case {'SGDM'}
            % <A stochastic approximation method>
            % <On the momentum term in gradient descent learning algorithms>
            mea = cellfun(@(deltaA, mA)b1*mA+deltaA, delta, mea, 'UniformOutput', false);
            param = cellfun(@(A, mA)A-lr*mA, param, mea, 'UniformOutput', false);
        case {'Adam'}
            % <Adam: A method for stochastic optimization>
            mea = cellfun(@(deltaA, mA)b1*mA+(1 - b1)*deltaA, delta, mea, 'UniformOutput', false);
            var = cellfun(@(deltaA, vA)b2*vA+(1 - b2)*deltaA.^2, delta, var, 'UniformOutput', false);
            mH = cellfun(@(mA)mA/(1 - b1^uit), mea, 'UniformOutput', false);
            vH = cellfun(@(vA)vA/(1 - b2^uit), var, 'UniformOutput', false);
            param = cellfun(@(A, mAH, vAH)A-lr*mAH./(sqrt(vAH) + sp), param, mH, vH, 'UniformOutput', false);
        case {'AdaBelief'}
            % <AdaBelief optimizer: Adapting stepsizes by the belief in observed gradients>
            mea = cellfun(@(deltaA, mA)b1*mA+(1 - b1)*deltaA, delta, mea, 'UniformOutput', false);
            tmp = cellfun(@(deltaA, mA)deltaA-mA, delta, mea, 'UniformOutput', false);
            var = cellfun(@(deltaA, vA)b2*vA+(1 - b2)*deltaA.^2, tmp, var, 'UniformOutput', false);
            mH = cellfun(@(mA)mA/(1 - b1^uit), mea, 'UniformOutput', false);
            vH = cellfun(@(vA)vA/(1 - b2^uit), var, 'UniformOutput', false);
            param = cellfun(@(A, mAH, vAH)A-lr*mAH./(sqrt(vAH) + sp), param, mH, vH, 'UniformOutput', false);
    end
    % unpackage trainable parameters (param) and derivatives (delta)
    switch DR % DimensionalityReduction
        case {'None', 'PCAfixed', 'RandFixed', 'PCAxyFixed'}
            [Antecedents, W] = deal(param{1}, param{2});
        case {'CDR', 'FS', 'CFS', 'RandInit', 'PCAxyInit', 'DR', 'FA1', 'FA2'}
            [Antecedents, W, WA] = deal(param{1}, param{2}, param{3});
            % Proximal derivative descent for L_{2,1} GroupLasso FeatureSelection
            if ismember(DR, {'FS', 'CFS'})
                tmp = max(vecnorm(WA(2:end, :), 2, 2)-FS*lr, 0);
                WA(2:end, :) = WA(2:end, :) ./ vecnorm(WA(2:end, :), 2, 2) .* tmp;
            end
        case {'CBN', 'BNA'}
            [Antecedents, W, gammaA, betaA] = deal(param{1}, param{2}, param{3}, param{4});
            % Package betaA and gammaA into WA
            WA = [betaA * ones(1, M); gammaA * eye(M)];
        case {'BNC'}
            [Antecedents, W, gammaC, betaC] = deal(param{1}, param{2}, param{3}, param{4});
            % Package betaC and gammaC into WC
            WC = [betaC * ones(1, M); gammaC * eye(M)];
        case {'FAx'}
            [Antecedents, W, WA, WC] = deal(param{1}, param{2}, param{3}, param{4});
    end
    if ismember(Init, {'FCM-LN', 'FCM-LN-ReLU'})
        [gammaL, betaL] = deal(param{end, -1}, param{end});
    end
    
    if ismember(it, iterationsRecorded)
        rit = rit + 1;
        % training result on the minibatch
        if isRegression
            % Root Mean Squared Error (RMSE) for regression, the lower the better
            trainResult(rit) = sqrt(mean((yReal-yPred).^2));
        else
            % Balanced Classification Accuracy (BCA) for classification, the higher the better
            [~, yPr] = max(yPred, [], 2);
            [~, yRe] = max(yReal, [], 2);
            CM = confusionmat(yRe, yPr);
            Sensitivity = diag(CM) ./ sum(CM, 2);
            trainResult(rit) = nanmean(Sensitivity); % Balanced Classification Accuracy (BCA)
            % mean(yRe == yPr); % Accuracy (ACC)
            % mean(diag(yReal*log(softmax(yPred')))); % Negative Cross Entropy (NCE)
        end
        % validation and test result
        for i = 1:length(testX)
            NTest = size(testX{i}, 1);
            testXbatch = testX{i};
            switch DR % DimensionalityReduction Layer Forward
                case {'None'}
                    [testXA, testXC] = deal(testXbatch);
                case {'PCAfixed', 'RandFixed', 'PCAxyFixed', 'CDR', 'FS', 'CFS', 'RandInit', 'PCAxyInit'}
                    [testXA, testXC] = deal([ones(NTest, 1), testXbatch]*WA);
                case {'DR'}
                    testXA = [ones(NTest, 1), testXbatch] * WA;
                    testXC = testXbatch;
                case {'CBN', 'BNA', 'BNC'}
                    % CBN: Consistant BatchNorm
                    % BNA: BatchNorm for Antecedents
                    % BNC: BatchNorm for Consequents
                    testXbn = (testXbatch - trainXmean) ./ trainXstd;
                    if ismember(DR, {'CBN'})
                        [testXA, testXC] = deal([ones(NTest, 1), testXbn]*WA);
                    elseif ismember(DR, {'BNA'})
                        testXA = [ones(NTest, 1), testXbn] * WA;
                        testXC = testXbatch;
                    elseif ismember(DR, {'BNC'})
                        testXA = testXbatch;
                        testXC = [ones(NTest, 1), testXbn] * WC;
                    end
                case {'FA1'} % FA1: FeatureAugmentation for Antecedents
                    testXA = [[ones(NTest, 1), testXbatch] * WA, testXbatch];
                    testXC = testXbatch;
                case {'FA2'} % FA2: Same FeatureAugmentation for Antecedents and Consequents
                    [testXA, testXC] = deal([[ones(NTest, 1), testXbatch] * WA, testXbatch]);
                case {'FAx'} % FAx: Different FeatureAugmentation for Antecedents and Consequents
                    testXA = [[ones(NTest, 1), testXbatch] * WA, testXbatch];
                    testXC = [[ones(NTest, 1), testXbatch] * WC, testXbatch];
            end
            % calculate Predictions on each rule
            tmp = permute(repmat(W, [1, 1, 1, NTest]), [4, 2, 1, 3]);
            yR = permute(sum([ones(NTest, 1), testXC] .* tmp, 2), [1, 3, 4, 2]);
            % Calculate FiringLevels using MembershipFunctions on each rule
            idsKeep = true(1, nRule);
            switch Uncertain
                case {'None'} % type-1 fuzzy set
                    fKeep = zeros(NTest, nRule); % firing level of rules
                    for n = 1:NTest
                        fKeep(n, :) = calculateFiringLevel(testXA(n, :), Antecedents, idsKeep, MF, nMF, Uncertain, Init);
                    end
                    if ismember(Init, {'LogFCM', 'LogHFCM'})
                        % LogFCM: logarithm transformed FiringLevel for high-dimensional input,
                        % following <A TSK-type convolutional recurrent fuzzy network for predicting driving fatigue>
                        fKeep = -1 ./ log(fKeep);
                    end
                    fBar = fKeep ./ sum(fKeep, 2);
                    if ismember(Init, {'FCM-LN', 'FCM-LN-ReLU'})
                        % LN: LayerNorm, a variant of BatchNorm,
                        % following <Layer normalization for TSK fuzzy system optimization in regression problems>
                        fBar = gammaL * zscore(fBar, 1, 2) + betaL;
                    end
                    testYPred = squeeze(sum(fBar .* yR, 2)); % averaged prediction
                case {'Mean', 'mean', 'Variance', 'var'} % interval type-2 fuzzy set
                    switch TR % TypeReduction
                        case {'Karnik-Mendel', 'km'}
                            % <Centroid of a type-2 fuzzy set>
                            fKeep = zeros(NTest, nRule, 2); % firing level of rules
                            testYPred = nan(NTest, nC);
                            for n = 1:NTest
                                fKeep(n, :, :) = calculateFiringLevel(testXA(n, :), Antecedents, idsKeep, MF, nMF, Uncertain, Init);
                                for c = 1:nC
                                    [syR, iyR] = sort(yR(n, :, c));
                                    sfl = fKeep(n, iyR, 1);
                                    sfr = fKeep(n, iyR, 2);
                                    % EIASC: Enhanced Iterative Algorithm with Stopping Condition,
                                    % <Comparison and practical implementation of type-reduction algorithms
                                    % for type-2 fuzzy sets and systems>
                                    % <A comprehensive study of the efficiency of type-reduction algorithms>
                                    testYPred(n, c) = EIASC(syR, syR, sfl, sfr, 0); % averaged prediction
                                end
                            end
                        case {'Nie-Tan'}
                            % <Towards an efficient typereduction method for interval type-2 fuzzy logic systems>
                            fKeep = zeros(NTest, nRule, 2); % firing level of rules
                            testYPred = nan(NTest, nC);
                            for n = 1:NTest
                                fKeep(n, :, :) = calculateFiringLevel(testXA(n, :), Antecedents, idsKeep, MF, nMF, Uncertain, Init);
                                fAvg = (fKeep(n, :, 1) + fKeep(n, :, 2)) / 2;
                                fBar = fAvg / sum(fAvg);
                                testYPred(n, :) = squeeze(sum(yR(n, :, :) .* fBar, 2)); % averaged prediction
                            end
                    end
            end
            if isRegression
                % Root Mean Squared Error (RMSE) for regression, the lower the better
                testResult{i}(rit) = sqrt(mean((testY{i}-testYPred).^2));
            else
                % Balanced Classification Accuracy (BCA) for classification, the higher the better
                [~, yPr] = max(testYPred, [], 2);
                [~, yRe] = max(testY{i}, [], 2);
                CM = confusionmat(yRe, yPr);
                Sensitivity = diag(CM) ./ sum(CM, 2);
                testResult{i}(rit) = nanmean(Sensitivity); % Balanced Classification Accuracy (BCA)
                % mean(yRe == yPr); % Accuracy (ACC)
                % mean(diag(testY{i}*log(softmax(testYPred')))); % Negative Cross Entropy (NCE)
            end
            % special case: testResult=nan;
            if isnan(testResult{i}(rit)) && rit > 1
                testResult{i}(rit) = testResult{i}(rit - 1);
            end
            % parameter validation
            if i == 1 && ((isRegression && testResult{i}(rit) < thre) || (~isRegression && testResult{i}(rit) > thre))
                thre = testResult{i}(rit);
                itB = rit;
                [AntecedentsB, WB, WAB, WCB, fB] = deal(Antecedents, W, WA, WC, mean(mean(fKeep, 3)));
                [gammaLB, betaLB] = deal(gammaL, betaL);
            end
        end
    end
end
if isDebug
    if isRegression
        fprintf("Iteration: %d, trainRMSE: %.2f, tuneRMSE: %.2f, testRMSE: %.2f.\n", ...
            iterationsRecorded(itB), trainResult(itB), testResult{1}(itB), testResult{2}(itB))
    else
        fprintf("Iteration: %d, trainBCA: %.2f, tuneBCA: %.2f, testBCA: %.2f.\n", ...
            iterationsRecorded(itB), trainResult(itB), testResult{1}(itB), testResult{2}(itB))
    end
end
% special case: validation set unavailable
if length(testX) == 1
    testResult = testResult{1};
end
tmp = {trainResult, testResult, AntecedentsB, WB, WAB, WCB, fB, gammaLB, betaLB};
varargout(1:nargout) = tmp(1:nargout);
end