function [threshold] = theta(s)
    threshold = exp(s) / (1 + exp(s));
end