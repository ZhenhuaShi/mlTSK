function [f, deltamu] = calculateFiringLevel(x, Antecedents, idsKeep, MF, nMF, Uncertain, Init)
minDenominator = 1e-3;
if ~nMF % independent MembershipFunctions
    Antecedents = Antecedents(idsKeep, :, :);
    switch Uncertain
        case {'None'} % type-1 fuzzy set
            switch MF % MembershipFunctions
                case {'Bell-Shaped', 'gbell'}
                    [A, B, C] = deal(squeeze(Antecedents(:, :, 1)), squeeze(Antecedents(:, :, 2)), ...
                        squeeze(Antecedents(:, :, 3)));
                    A = max(A, minDenominator);
                    f = prod(1./(1 + ((x - C) ./ A).^2.^B), 2);
                    if nargout > 1
                        deltaA = 2 * B .* (x - C).^2.^B .* A.^(2 * B - 1) ./ (A.^2.^B + (x - C).^2.^B).^2;
                        deltaB = -(x - C).^2.^B .* A.^2.^B ./ (A.^2.^B + (x - C).^2.^B).^2 .* log(((x - C)./A).^2);
                        deltaC = 2 * B .* (x - C).^2.^B .* A.^2.^B ./ (x - C) ./ (A.^2.^B + (x - C).^2.^B).^2;
                        deltamu = cat(3, deltaA, deltaB, deltaC);
                    end
                case {'Gaussian', 'gauss'}
                    [Sigma, C] = deal(squeeze(Antecedents(:, :, 1)), squeeze(Antecedents(:, :, 2)));
                    Sigma = max(Sigma, minDenominator);
                    if ismember(Init, {'LogFCM', 'LogHFCM'})
                        f = exp(sum(-(x - C).^2 ./ (2 * Sigma.^2), 2));
                    else
                        f = softmax(sum(-(x - C).^2 ./ (2 * Sigma.^2), 2));
                    end
                    if nargout > 1
                        deltaC = (x - C) ./ (Sigma.^2);
                        deltaSigma = (x - C).^2 ./ (Sigma.^3);
                        deltamu = cat(3, deltaSigma, deltaC);
                    end
                case {'Trapezoidal', 'trap', 'Triangular', 'tri'}
                    if ~issorted(Antecedents, 3)
                        Antecedents = sort(Antecedents, 3);
                    end
                    switch MF % MembershipFunctions
                        case {'Trapezoidal', 'trap'}
                            [A, B, C, D] = deal(squeeze(Antecedents(:, :, 1)), squeeze(Antecedents(:, :, 2)), ...
                                squeeze(Antecedents(:, :, 3)), squeeze(Antecedents(:, :, 4)));
                        case {'Triangular', 'tri'}
                            [A, B, C, D] = deal(squeeze(Antecedents(:, :, 1)), squeeze(Antecedents(:, :, 2)), ...
                                squeeze(Antecedents(:, :, 2)), squeeze(Antecedents(:, :, 3)));
                    end
                    a1 = (x - A) ./ (B - A);
                    a2 = (D - x) ./ (D - C);
                    mu = max(0, min(1, min(a1, a2)));
                    f = prod(mu, 2);
                    if nargout > 1
                        [deltaA, deltaB, deltaC, deltaD] = deal(zeros(size(A)));
                        X = x .* ones(size(A));
                        id1 = (a1 > 0) + (a1 < 1) == 2;
                        if sum(id1(:))
                            deltaA(id1) = (X(id1) - B(id1)) ./ (a1(id1) .* (B(id1) - A(id1)).^2);
                            deltaB(id1) = -1 ./ (B(id1) - A(id1));
                        end
                        id2 = (a2 > 0) + (a2 < 1) == 2;
                        if sum(id2(:))
                            deltaD(id2) = (X(id2) - C(id2)) ./ (a2(id2) .* (C(id2) - D(id2)).^2);
                            deltaC(id2) = -1 ./ (C(id2) - D(id2));
                        end
                        switch MF % MembershipFunctions
                            case {'Trapezoidal', 'trap'}
                                deltamu = cat(3, deltaA, deltaB, deltaC, deltaD);
                            case {'Triangular', 'tri'}
                                deltamu = cat(3, deltaA, deltaB+deltaC, deltaD);
                        end
                    end
            end
        case {'Mean', 'mean'} % interval type-2 fuzzy set with uncertain mean
            switch MF % MembershipFunctions
                case {'Gaussian', 'gauss'}
                    [Sigma, Cl, Cr] = deal(squeeze(Antecedents(:, :, 1)), squeeze(Antecedents(:, :, 2)), ...
                        squeeze(Antecedents(:, :, 3)));
                    Sigma = max(Sigma, minDenominator);
                    ids = Cl > Cr;
                    [Cl(ids), Cr(ids)] = deal(Cr(ids), Cl(ids));
                    mul = -(x - (Cr + (x > (Cl + Cr) / 2) .* (Cl - Cr))).^2 ./ (2 * Sigma.^2);
                    fl = exp(sum(mul, 2));
                    mur = -(x - (Cl + (x > Cl) .* (Cr - Cl + (x < Cr) .* (x - Cr)))).^2 ./ (2 * Sigma.^2);
                    fr = exp(sum(mur, 2));
                    f = [fl, fr];
                    if nargout > 1
                        deltaCl = permute(cat(3, (x - (x + (x > (Cl + Cr) / 2) .* Cl)) ./ (Sigma.^2), ...
                            (x - (Cl + (x > Cl) .* (x - Cl))) ./ (Sigma.^2)), [1, 3, 2]);
                        deltaCr = permute(cat(3, (x - (Cr + (x > (Cl + Cr) / 2) .* (x - Cr))) ./ (Sigma.^2), ...
                            (x - (Cr + (x < Cr) .* (x - Cr))) ./ (Sigma.^2)), [1, 3, 2]);
                        deltaSigma = permute(cat(3, (x - (Cr + (x > (Cl + Cr) / 2) .* (Cl - Cr))).^2 ./ (Sigma.^3), ...
                            (x - (Cl + (x > Cl) .* (Cr - Cl + (x < Cr) .* (x - Cr)))).^2 ./ (Sigma.^3)), [1, 3, 2]);
                        deltamu = cat(4, deltaSigma, deltaCl, deltaCr);
                    end
            end
        case {'Variance', 'var'} % interval type-2 fuzzy set with uncertain variance
            switch MF % MembershipFunctions
                case {'Gaussian', 'gauss'}
                    [Sigmal, Sigmar, C] = deal(squeeze(Antecedents(:, :, 1)), squeeze(Antecedents(:, :, 2)), ...
                        squeeze(Antecedents(:, :, 3)));
                    Sigmal = max(Sigmal, minDenominator);
                    Sigmar = max(Sigmar, minDenominator);
                    ids = Sigmal > Sigmar;
                    [Sigmal(ids), Sigmar(ids)] = deal(Sigmar(ids), Sigmal(ids));
                    mul = -(x - C).^2 ./ (2 * Sigmal.^2);
                    fl = exp(sum(mul, 2));
                    mur = -(x - C).^2 ./ (2 * Sigmar.^2);
                    fr = exp(sum(mur, 2));
                    f = [fl, fr];
                    if nargout > 1
                        deltaC = permute(cat(3, (x - C) ./ (Sigmal.^2), (x - C) ./ (Sigmar.^2)), [1, 3, 2]);
                        deltaSigmal = permute(cat(3, (x - C).^2 ./ (Sigmal.^3), zeros(size(Sigmal))), [1, 3, 2]);
                        deltaSigmar = permute(cat(3, zeros(size(Sigmar)), (x - C).^2 ./ (Sigmar.^3)), [1, 3, 2]);
                        deltamu = cat(4, deltaSigmal, deltaSigmar, deltaC);
                    end
            end
    end
else % shared MembershipFunctions
    switch Uncertain
        case {'None'} % type-1 fuzzy set
            switch MF % MembershipFunctions
                case {'Bell-Shaped', 'gbell'}
                    [A, B, C] = deal(squeeze(Antecedents(:, :, 1)), squeeze(Antecedents(:, :, 2)), ...
                        squeeze(Antecedents(:, :, 3)));
                    A = max(A, minDenominator);
                    B = max(1, B);
                    mu = 1 ./ (1 + ((x' - C) ./ A).^2.^B);
                    if nargout > 1
                        deltaA = 2 * B .* (x' - C).^2.^B .* A.^(2 * B - 1) ./ (A.^2.^B + (x' - C).^2.^B).^2;
                        deltaB = -(x' - C).^2.^B .* A.^2.^B ./ (A.^2.^B + (x' - C).^2.^B).^2 .* log(((x' - C)./A).^2+eps);
                        deltaC = 2 * B .* (x' - C).^2.^B .* A.^2.^B ./ (x' - C + eps) ./ (A.^2.^B + (x' - C).^2.^B).^2;
                    end
                    for m = 1:size(x, 2) % membership grades of MFs
                        if m == 1
                            pmu = mu(m, :);
                            if nargout > 1
                                [deltapA, deltapB, deltapC] = deal(zeros(1, nMF, nMF));
                                deltapA(1, :, :) = diag(deltaA(m, :));
                                deltapB(1, :, :) = diag(deltaB(m, :));
                                deltapC(1, :, :) = diag(deltaC(m, :));
                            end
                        else
                            pmu = [repmat(pmu, 1, nMF); reshape(repmat(mu(m, :), size(pmu, 2), 1), 1, [])];
                            if nargout > 1
                                deltapA = [repmat(deltapA, 1, nMF); permute(reshape(repmat(diag(deltaA(m, :)), ...
                                    size(deltapA, 2), 1), nMF, []), [3, 2, 1])];
                                deltapB = [repmat(deltapB, 1, nMF); permute(reshape(repmat(diag(deltaB(m, :)), ...
                                    size(deltapB, 2), 1), nMF, []), [3, 2, 1])];
                                deltapC = [repmat(deltapC, 1, nMF); permute(reshape(repmat(diag(deltaC(m, :)), ...
                                    size(deltapC, 2), 1), nMF, []), [3, 2, 1])];
                            end
                        end
                    end
                    f = prod(pmu(:, idsKeep), 1)';
                    if nargout > 1
                        deltamu = cat(4, deltapA(:, idsKeep, :), deltapB(:, idsKeep, :), deltapC(:, idsKeep, :));
                    end
                case {'Gaussian', 'gauss'}
                    [Sigma, C] = deal(squeeze(Antecedents(:, :, 1)), squeeze(Antecedents(:, :, 2)));
                    mu = -(x' - C).^2 ./ (2 * Sigma.^2);
                    if nargout > 1
                        deltaC = (x' - C) ./ (Sigma.^2);
                        deltaSigma = (x' - C).^2 ./ (Sigma.^3);
                    end
                    for m = 1:size(x, 2) % membership grades of MFs
                        if m == 1
                            pmu = mu(m, :);
                            if nargout > 1
                                [deltapC, deltapSigma] = deal(zeros(1, nMF, nMF));
                                deltapC(1, :, :) = diag(deltaC(m, :));
                                deltapSigma(1, :, :) = diag(deltaSigma(m, :));
                            end
                        else
                            pmu = [repmat(pmu, 1, nMF); reshape(repmat(mu(m, :), size(pmu, 2), 1), 1, [])];
                            if nargout > 1
                                deltapC = [repmat(deltapC, 1, nMF); ...
                                    permute(reshape(repmat(diag(deltaC(m, :)), size(deltapC, 2), 1), nMF, []), [3, 2, 1])];
                                deltapSigma = [repmat(deltapSigma, 1, nMF); ...
                                    permute(reshape(repmat(diag(deltaSigma(m, :)), size(deltapSigma, 2), 1), nMF, []), [3, 2, 1])];
                            end
                        end
                    end
                    if ismember(Init, {'LogFCM', 'LogHFCM'})
                        f = exp(sum(pmu(:, idsKeep), 1)');
                    else
                        f = softmax(sum(pmu(:, idsKeep), 1)');
                    end
                    if nargout > 1
                        deltamu = cat(4, deltapSigma(:, idsKeep, :), deltapC(:, idsKeep, :));
                    end
                case {'Trapezoidal', 'trap', 'Triangular', 'tri'}
                    if ~issorted(Antecedents, 3)
                        Antecedents = sort(Antecedents, 3);
                    end
                    switch MF % MembershipFunctions
                        case {'Trapezoidal', 'trap'}
                            [A, B, C, D] = deal(squeeze(Antecedents(:, :, 1)), squeeze(Antecedents(:, :, 2)), ...
                                squeeze(Antecedents(:, :, 3)), squeeze(Antecedents(:, :, 4)));
                        case {'Triangular', 'tri'}
                            [A, B, C, D] = deal(squeeze(Antecedents(:, :, 1)), squeeze(Antecedents(:, :, 2)), ...
                                squeeze(Antecedents(:, :, 2)), squeeze(Antecedents(:, :, 3)));
                    end
                    a1 = (x' - A) ./ (B - A);
                    a2 = (D - x') ./ (D - C);
                    mu = max(0, min(1, min(a1, a2)));
                    if nargout > 1
                        [deltaA, deltaB, deltaC, deltaD] = deal(zeros(size(A)));
                        X = x' .* ones(size(A));
                        id1 = (a1 > 0) + (a1 < 1) == 2;
                        if sum(id1(:))
                            deltaA(id1) = (X(id1) - B(id1)) ./ (a1(id1) .* (B(id1) - A(id1)).^2);
                            deltaB(id1) = -1 ./ (B(id1) - A(id1));
                        end
                        id2 = (a2 > 0) + (a2 < 1) == 2;
                        if sum(id2(:))
                            deltaD(id2) = (X(id2) - C(id2)) ./ (a2(id2) .* (C(id2) - D(id2)).^2);
                            deltaC(id2) = -1 ./ (C(id2) - D(id2));
                        end
                    end
                    for m = 1:size(x, 2) % membership grades of MFs
                        if m == 1
                            pmu = mu(m, :);
                            if nargout > 1
                                [deltapA, deltapB, deltapC, deltapD] = deal(zeros(1, nMF, nMF));
                                deltapA(1, :, :) = diag(deltaA(m, :));
                                deltapB(1, :, :) = diag(deltaB(m, :));
                                deltapC(1, :, :) = diag(deltaC(m, :));
                                deltapD(1, :, :) = diag(deltaD(m, :));
                            end
                        else
                            pmu = [repmat(pmu, 1, nMF); reshape(repmat(mu(m, :), size(pmu, 2), 1), 1, [])];
                            if nargout > 1
                                deltapA = [repmat(deltapA, 1, nMF); ...
                                    permute(reshape(repmat(diag(deltaA(m, :)), size(deltapA, 2), 1), nMF, []), [3, 2, 1])];
                                deltapB = [repmat(deltapB, 1, nMF); ...
                                    permute(reshape(repmat(diag(deltaB(m, :)), size(deltapB, 2), 1), nMF, []), [3, 2, 1])];
                                deltapC = [repmat(deltapC, 1, nMF); ...
                                    permute(reshape(repmat(diag(deltaC(m, :)), size(deltapC, 2), 1), nMF, []), [3, 2, 1])];
                                deltapD = [repmat(deltapD, 1, nMF); ...
                                    permute(reshape(repmat(diag(deltaD(m, :)), size(deltapD, 2), 1), nMF, []), [3, 2, 1])];
                            end
                        end
                    end
                    f = prod(pmu(:, idsKeep), 1)';
                    if nargout > 1
                        switch MF % MembershipFunctions
                            case {'Trapezoidal', 'trap'}
                                deltamu = cat(4, deltapA(:, idsKeep, :), ...
                                    deltapB(:, idsKeep, :), deltapC(:, idsKeep, :), deltapD(:, idsKeep, :));
                            case {'Triangular', 'tri'}
                                deltamu = cat(4, deltapA(:, idsKeep, :), ...
                                    deltapB(:, idsKeep, :)+deltapC(:, idsKeep, :), deltapD(:, idsKeep, :));
                        end
                    end
            end
        case {'Mean', 'mean'} % interval type-2 fuzzy set with uncertain mean
            switch MF % MembershipFunctions
                case {'Gaussian', 'gauss'}
                    [Sigma, Cl, Cr] = deal(squeeze(Antecedents(:, :, 1)), squeeze(Antecedents(:, :, 2)), ...
                        squeeze(Antecedents(:, :, 3)));
                    Sigma = max(Sigma, minDenominator);
                    ids = Cl > Cr;
                    [Cl(ids), Cr(ids)] = deal(Cr(ids), Cl(ids));
                    mul = -(x' - (Cr + (x' > (Cl + Cr) / 2) .* (Cl - Cr))).^2 ./ (2 * Sigma.^2);
                    mur = -(x' - (Cl + (x' > Cl) .* (Cr - Cl + (x' < Cr) .* (x' - Cr)))).^2 ./ (2 * Sigma.^2);
                    if nargout > 1
                        deltaCl = permute(cat(3, (x' - (x' + (x' > (Cl + Cr) / 2) .* Cl)) ./ (Sigma.^2), ...
                            (x' - (Cl + (x' > Cl) .* (x' - Cl))) ./ (Sigma.^2)), [1, 3, 2]);
                        deltaCr = permute(cat(3, (x' - (Cr + (x' > (Cl + Cr) / 2) .* (x' - Cr))) ./ (Sigma.^2), ...
                            (x' - (Cr + (x' < Cr) .* (x' - Cr))) ./ (Sigma.^2)), [1, 3, 2]);
                        deltaSigma = permute(cat(3, (x' - (Cr + (x' > (Cl + Cr) / 2) .* (Cl - Cr))).^2 ./ (Sigma.^3), ...
                            (x' - (Cl + (x' > Cl) .* (Cr - Cl + (x' < Cr) .* (x' - Cr)))).^2 ./ (Sigma.^3)), [1, 3, 2]);
                    end
                    for m = 1:size(x, 2) % membership grades of MFs
                        if m == 1
                            pmul = mul(m, :);
                            pmur = mur(m, :);
                            if nargout > 1
                                [deltaplCl, deltaprCl, deltaplCr, deltaprCr, deltaplSigma, deltaprSigma] ...
                                    = deal(zeros(1, nMF, nMF));
                                deltaplCl(1, :, :) = diag(squeeze(deltaCl(m, 1, :)));
                                deltaprCl(1, :, :) = diag(squeeze(deltaCl(m, 2, :)));
                                deltaplCr(1, :, :) = diag(squeeze(deltaCr(m, 1, :)));
                                deltaprCr(1, :, :) = diag(squeeze(deltaCr(m, 2, :)));
                                deltaplSigma(1, :, :) = diag(squeeze(deltaSigma(m, 1, :)));
                                deltaprSigma(1, :, :) = diag(squeeze(deltaSigma(m, 2, :)));
                            end
                        else
                            pmul = [repmat(pmul, 1, nMF); reshape(repmat(mul(m, :), size(pmul, 2), 1), 1, [])];
                            pmur = [repmat(pmur, 1, nMF); reshape(repmat(mur(m, :), size(pmur, 2), 1), 1, [])];
                            if nargout > 1
                                tmp = size(deltaplCl, 2);
                                deltaplCl = [repmat(deltaplCl, 1, nMF); ...
                                    permute(reshape(repmat(diag(squeeze(deltaCl(m, 1, :))), tmp, 1), nMF, []), [3, 2, 1])];
                                deltaprCl = [repmat(deltaprCl, 1, nMF); ...
                                    permute(reshape(repmat(diag(squeeze(deltaCl(m, 2, :))), tmp, 1), nMF, []), [3, 2, 1])];
                                deltaplCr = [repmat(deltaplCr, 1, nMF); ...
                                    permute(reshape(repmat(diag(squeeze(deltaCr(m, 1, :))), tmp, 1), nMF, []), [3, 2, 1])];
                                deltaprCr = [repmat(deltaprCr, 1, nMF); ...
                                    permute(reshape(repmat(diag(squeeze(deltaCr(m, 2, :))), tmp, 1), nMF, []), [3, 2, 1])];
                                deltaplSigma = [repmat(deltaplSigma, 1, nMF); ...
                                    permute(reshape(repmat(diag(squeeze(deltaSigma(m, 1, :))), tmp, 1), nMF, []), [3, 2, 1])];
                                deltaprSigma = [repmat(deltaprSigma, 1, nMF); ...
                                    permute(reshape(repmat(diag(squeeze(deltaSigma(m, 2, :))), tmp, 1), nMF, []), [3, 2, 1])];
                            end
                        end
                    end
                    fl = exp(sum(pmul(:, idsKeep), 1)');
                    fr = exp(sum(pmur(:, idsKeep), 1)');
                    f = [fl, fr];
                    if nargout > 1
                        deltapCl = permute(cat(4, deltaplCl(:, idsKeep, :), deltaprCl(:, idsKeep, :)), [1, 2, 4, 3]);
                        deltapCr = permute(cat(4, deltaplCr(:, idsKeep, :), deltaprCr(:, idsKeep, :)), [1, 2, 4, 3]);
                        deltapSigma = permute(cat(4, deltaplSigma(:, idsKeep, :), deltaprSigma(:, idsKeep, :)), [1, 2, 4, 3]);
                        deltamu = cat(5, deltapSigma, deltapCl, deltapCr);
                    end
            end
        case {'Variance', 'var'} % interval type-2 fuzzy set with uncertain variance
            switch MF % MembershipFunctions
                case {'Gaussian', 'gauss'}
                    [Sigmal, Sigmar, C] = deal(squeeze(Antecedents(:, :, 1)), squeeze(Antecedents(:, :, 2)), ...
                        squeeze(Antecedents(:, :, 3)));
                    Sigmal = max(Sigmal, minDenominator);
                    Sigmar = max(Sigmar, minDenominator);
                    ids = Sigmal > Sigmar;
                    [Sigmal(ids), Sigmar(ids)] = deal(Sigmar(ids), Sigmal(ids));
                    mul = -(x' - C).^2 ./ (2 * Sigmal.^2);
                    mur = -(x' - C).^2 ./ (2 * Sigmar.^2);
                    if nargout > 1
                        deltaC = permute(cat(3, (x' - C) ./ (Sigmal.^2), (x' - C) ./ (Sigmar.^2)), [1, 3, 2]);
                        deltaSigmal = permute(cat(3, (x' - C).^2 ./ (Sigmal.^3), zeros(size(Sigmal))), [1, 3, 2]);
                        deltaSigmar = permute(cat(3, zeros(size(Sigmar)), (x' - C).^2 ./ (Sigmar.^3)), [1, 3, 2]);
                    end
                    for m = 1:size(x, 2) % membership grades of MFs
                        if m == 1
                            pmul = mul(m, :);
                            pmur = mur(m, :);
                            if nargout > 1
                                [deltaplC, deltaprC, deltaplSigmal, deltaprSigmal, deltaplSigmar, deltaprSigmar] ...
                                    = deal(zeros(1, nMF, nMF));
                                deltaplC(1, :, :) = diag(squeeze(deltaC(m, 1, :)));
                                deltaprC(1, :, :) = diag(squeeze(deltaC(m, 2, :)));
                                deltaplSigmal(1, :, :) = diag(squeeze(deltaSigmal(m, 1, :)));
                                deltaprSigmal(1, :, :) = diag(squeeze(deltaSigmal(m, 2, :)));
                                deltaplSigmar(1, :, :) = diag(squeeze(deltaSigmar(m, 1, :)));
                                deltaprSigmar(1, :, :) = diag(squeeze(deltaSigmar(m, 2, :)));
                            end
                        else
                            pmul = [repmat(pmul, 1, nMF); reshape(repmat(mul(m, :), size(pmul, 2), 1), 1, [])];
                            pmur = [repmat(pmur, 1, nMF); reshape(repmat(mur(m, :), size(pmur, 2), 1), 1, [])];
                            if nargout > 1
                                tmp = size(deltaplC, 2);
                                deltaplC = [repmat(deltaplC, 1, nMF); ...
                                    permute(reshape(repmat(diag(squeeze(deltaC(m, 1, :))), tmp, 1), nMF, []), [3, 2, 1])];
                                deltaprC = [repmat(deltaprC, 1, nMF); ...
                                    permute(reshape(repmat(diag(squeeze(deltaC(m, 2, :))), tmp, 1), nMF, []), [3, 2, 1])];
                                tmp = size(deltaprSigmal, 2);
                                deltaplSigmal = [repmat(deltaplSigmal, 1, nMF); ...
                                    permute(reshape(repmat(diag(squeeze(deltaSigmal(m, 1, :))), tmp, 1), nMF, []), [3, 2, 1])];
                                deltaprSigmal = [repmat(deltaprSigmal, 1, nMF); ...
                                    permute(reshape(repmat(diag(squeeze(deltaSigmal(m, 2, :))), tmp, 1), nMF, []), [3, 2, 1])];
                                deltaplSigmar = [repmat(deltaplSigmar, 1, nMF); ...
                                    permute(reshape(repmat(diag(squeeze(deltaSigmar(m, 1, :))), tmp, 1), nMF, []), [3, 2, 1])];
                                deltaprSigmar = [repmat(deltaprSigmar, 1, nMF); ...
                                    permute(reshape(repmat(diag(squeeze(deltaSigmar(m, 2, :))), tmp, 1), nMF, []), [3, 2, 1])];
                            end
                        end
                    end
                    fl = exp(sum(pmul(:, idsKeep), 1)');
                    fr = exp(sum(pmur(:, idsKeep), 1)');
                    f = [fl, fr];
                    if nargout > 1
                        deltapC = permute(cat(4, deltaplC(:, idsKeep, :), deltaprC(:, idsKeep, :)), [1, 2, 4, 3]);
                        deltapSigmal = permute(cat(4, deltaplSigmal(:, idsKeep, :), deltaprSigmal(:, idsKeep, :)), [1, 2, 4, 3]);
                        deltapSigmar = permute(cat(4, deltaplSigmar(:, idsKeep, :), deltaprSigmar(:, idsKeep, :)), [1, 2, 4, 3]);
                        deltamu = cat(5, deltapSigmal, deltapSigmar, deltapC);
                    end
            end
    end
end
end