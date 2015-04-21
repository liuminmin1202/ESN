function teachCollectMat = compute_teacher_for_CNT(outputSequence, paramStruct, ...
    nForgetPoints)
% COMPUTE_TEACHER scales, shifts and applies the inverse output
% activation function on the exepcted teacher. 
% the first nForgetPoints are being disregarded
%
% inputs:
% outputSequence = teacher vector of size nTrainingPoints x nOutputDimension 
% nForgetPoints: an integer, may be negative, positive or zero.
%    If positive: the first nForgetPoints will be disregarded (washing out
%    initial reservoir transient)
%    If negative: the network will be initially driven from zero state with
%    the first input repeated |nForgetPoints| times; size(inputSequence,1)
%    many states will be sorted into state matrix
%    If zero: no washout accounted for, all states except the zero starting
%    state will be sorted into state matrix
%
% outputs:
% teachCollectMat = matrix of size (nOutputPoints - nForgetPoints) x
% nOutputUnits
% teachCollectMat contains the shifted and scaled output

%
% Version 1.0, April 30, 2006
% Copyright: Fraunhofer IAIS 2006 / Patent pending
% Revision 1, June 7, 2006, H.Jaeger

% Revision (for physical systems) 2.0, April 21, 2015
% Author: Matt Dale
% inputs: 
% paramMatrix = contains the information about the
% transformation we need to apply to the teacher


nOutputPoints  = length(outputSequence(:,1)) ; 
teachCollectMat = zeros(nOutputPoints - max([0, nForgetPoints]), paramStruct.nOutputUnits) ;

% delete the first nForgetPoints elements from outputSequence
if nForgetPoints >= 0
    outputSequence = outputSequence(nForgetPoints+1:end,:) ; 
end

% update the size of outputSequence
nOutputPoints  = length(outputSequence(:,1)) ; 

teachCollectMat = [(diag(paramStruct.teacherScaling) * outputSequence')' + ...
        repmat(paramStruct.teacherShift',[nOutputPoints 1])];

teachCollectMat = feval('identity', teachCollectMat);




