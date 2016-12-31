function [Ein w b] = LogisticRegression(Data, Rate)
    % Get number of datapoints
    N = size(Data, 1);
    % Set the maximum number of iterations
    MaxIterations = 10000000;
    IterationCount = 0;
    
    % initialize w as random normally distributed numbers with mean 0 and
    % variance 1
    w = randn(3, 1);
    
    oldmag = 0;
    mag = Inf;
    
    % initialize thingybingy
    Ein = [];
    
    while (IterationCount < MaxIterations) && (mag > 0.001) && (abs(oldmag - mag) > 0.00000001)
       IterationCount = IterationCount + 1;
        
       
       wT = transpose(w);
       Xs = [Data(:, [1 2]) ones(N, 1)];
       Ys = Data(:, 3);
       
       % compute the gradient
       Sum = 0;
       for n = 1 : N
           y = Ys(n);
           x = transpose(Xs(n, :));
           Sum = Sum + (-y * x * theta(-y * wT *x));
       end
       g = 1/N * Sum;
       
       % We update w
       w = w - Rate*g;
       oldmag = mag;
       mag = sqrt(sum(g.*g));
       
       % Update E
       err = 0;
       for n = 1 : N
          err = err + log(1+ exp(-y*transpose(w)*x));
       end
       err = 1/N * err;
       Ein = [Ein err];
    end
    b = w(3);
    w = w([1 2], :);
end



