function [out] = perceptron(in, threshold)

if in > threshold
    
    out = 1;
    
else
    
    out = in;

end