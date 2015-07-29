%% Create a simple 16x16 grid of spatially-defined internal weights
% used as an experiment to try and better define internal weights to some
% spatial dimensionality/representation

weights = zeros(16,16);

for i = 1:16
    for j = 1:16
   
        if j == i+1 || j== i-1 || j == i+4 || j== i-4 || j == i+3 || j== i+5 || j == i-3 || j== i-5
            if (mod(i,4) == 0 && (j == i+1))  ||  (mod(i-1,4) == 0 && (j == i-1)) || (mod(i,4) == 0 && (j == i-3))  ||  (mod(i+3,4) == 0 && (j == i+3))|| (mod(i,4) == 0 && (j == i+5))  ||  (mod(i-5,4) == 0 && (j == i-5))
                weights(i,j) = 0;
            else
                weights(i,j) = rand;
            end
        else
            weights(i,j) = 0;
        end
    end
end