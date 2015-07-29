% mutate matrix 
function [outMatrix]= mutate(inMatrix, perMutate)

outMatrix = inMatrix;
numMutate = round((length(inMatrix)/100)*perMutate); % percent mutate

for i = 1:numMutate
    pos =  randi([1 length(inMatrix)]);
    if rand >= 0.5
        outMatrix(pos) = 2*rand-1; %only between -1 and 1
    else
        outMatrix(pos) = 0;
    end
end