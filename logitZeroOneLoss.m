function [zeroOneLoss] = logitZeroOneLoss(w, b, Data)
    N = size(Data, 1);
    Xs = transpose(Data(:, [1 2]));
    Ys = Data(:, 3);
    Labels = zeros(N, 1);
    wT = transpose(w);
    
    for n = 1 : N
       Labels(n) = 2*(round(theta(wT * Xs(:, n) + b)) - 0.5);
    end
    
    zeroOneLoss = 0;
    for n = 1 : N
        zeroOneLoss = zeroOneLoss + (Labels(n) ~= Ys(n));
    end
    zeroOneLoss = zeroOneLoss / N;
end