
clear all; 
rand('seed', 10);
%% Create reservoir
%scale inputs and teacher attributes
nInputUnits = 1; nInternalUnits = 16; nOutputUnits = 1; 
numElectrodes = 2; 
nForgetPoints = 100; % discard the first 100 points
sequenceLength = 1000;
inputScale = zeros(nInputUnits,1);
inputShift = zeros(nInputUnits,1);

for i = 1:nInputUnits
    inputScale(i,:) = 1; %0.3
    inputShift(i,:) = 1;%-0.2 necessary to correlate input and prediction/target
end

teacherScaling = zeros(nOutputUnits,1); teacherShift = zeros(nOutputUnits,1);

for i = 1:nOutputUnits
    teacherScaling(i,:) = 1;%0.3;
    teacherShift(i,:) = 1;%-0.2;
end

%% Create reservoir with correct scaling - best specrad 0.8948
esn = generate_esn(nInputUnits, nInternalUnits, nOutputUnits, ...
    'spectralRadius',0.5,'inputScaling',inputScale,'inputShift',inputShift, ...
    'teacherScaling',teacherScaling,'teacherShift',teacherShift,'feedbackScaling', 0, ...
    'type', 'plain_esn');


esn2 = generate_esn(nInputUnits, nInternalUnits, nOutputUnits, ...
    'spectralRadius',0.1,'inputScaling',inputScale,'inputShift',inputShift, ...
    'teacherScaling',teacherScaling,'teacherShift',teacherShift,'feedbackScaling', 0, ...
    'type', 'plain_esn');

esn3 = generate_esn(nInputUnits, nInternalUnits, nOutputUnits, ...
    'spectralRadius',1,'inputScaling',inputScale,'inputShift',inputShift, ...
    'teacherScaling',teacherScaling,'teacherShift',teacherShift,'feedbackScaling', 0, ...
    'type', 'plain_esn');


esn4 = generate_esn(nInputUnits, nInternalUnits, nOutputUnits, ...
    'spectralRadius',1.5,'inputScaling',inputScale,'inputShift',inputShift, ...
    'teacherScaling',teacherScaling,'teacherShift',teacherShift,'feedbackScaling', 0, ...
    'type', 'plain_esn');


esn5 = generate_esn(nInputUnits, nInternalUnits, nOutputUnits, ...
    'spectralRadius',0.8,'inputScaling',inputScale,'inputShift',inputShift, ...
    'teacherScaling',teacherScaling,'teacherShift',teacherShift,'feedbackScaling', 0, ...
    'type', 'plain_esn');
%esn.inputWeights = ones(nInternalUnits, nInputUnits);
%% Assign input data and collect target output 
T = 10*(1/10);
Fs = 1000;
dt = 1/Fs;
t = 0:dt:T-dt;
%amplitude
A=0.5; %between 0-15

% Define input sequence
trainInputSequence(1,:)= A*sin(2*pi*10*t);

% configuration voltages
%trainInputSequence(2,:)= 0;%A*cos(2*pi*20*t);
%trainInputSequence(3,:)= 0;%A*square(2*pi*20*t);

% Desired output
trainOutputSequence(1,:) = A*sawtooth(2*pi*10*t);
%trainOutputSequence(1,:) = A*sin(2*pi*20*t);

% Multiple outputs
% len = 1;
% for i =1:len
%     trainOutputSequence(((i-1)*len)+1,:) = A*square(2*pi*10*t);
%     trainOutputSequence(((i-1)*len)+2,:) = A*cos(2*pi*10*t);
%     trainOutputSequence(((i-1)*len)+3,:) = A*sin(2*pi*20*t);
%     trainOutputSequence(((i-1)*len)+4,:) = A*sawtooth(2*pi*10*t);
%     
%     %x = sawtooth(2*pi*10*t);
% end

%% Split training set
train_fraction = 0.5 ; % use 50% in training and 50% in testing
[trainInputSequence,testInputSequence ] = ...
    split_train_test(trainInputSequence',train_fraction);
[trainOutputSequence,testOutputSequence ] = ...
    split_train_test(trainOutputSequence',train_fraction);

%% train reservoir
%scale the reservoir using the 'spectral radius' i.e. absolute eigenvector
esn.internalWeights = esn.spectralRadius * esn.internalWeights_UnitSR;

esn2.internalWeights = esn2.spectralRadius * esn2.internalWeights_UnitSR;
esn3.internalWeights = esn3.spectralRadius * esn3.internalWeights_UnitSR;
esn4.internalWeights = esn4.spectralRadius * esn4.internalWeights_UnitSR;
esn5.internalWeights = esn5.spectralRadius * esn5.internalWeights_UnitSR;
% % Train network  
% [trainedEsn, stateCollection] = ...
%     train_esn(trainInputSequence, trainOutputSequence , esn, nForgetPoints);
% 
% % Collect output from trained network 
% predictedTrainOutput = test_esn(trainInputSequence, trainedEsn, nForgetPoints);

stateCollection = compute_statematrix(trainInputSequence, trainOutputSequence, esn, nForgetPoints) ;
teacherCollection = compute_teacher(trainOutputSequence, esn, nForgetPoints) ;

stateCollection2 = compute_statematrix(trainInputSequence, trainOutputSequence, esn2, nForgetPoints) ;
teacherCollection2 = compute_teacher(trainOutputSequence, esn2, nForgetPoints) ;

stateCollection3 = compute_statematrix(trainInputSequence, trainOutputSequence, esn3, nForgetPoints) ;
teacherCollection3 = compute_teacher(trainOutputSequence, esn3, nForgetPoints) ;

stateCollection4 = compute_statematrix(trainInputSequence, trainOutputSequence, esn4, nForgetPoints) ;
teacherCollection4 = compute_teacher(trainOutputSequence, esn4, nForgetPoints) ;

stateCollection5 = compute_statematrix(trainInputSequence, trainOutputSequence, esn5, nForgetPoints) ;
teacherCollection5 = compute_teacher(trainOutputSequence, esn5, nForgetPoints) ;

start = 1;
endp = 10;
new_stateCollection = [stateCollection(:,start:endp) stateCollection2(:,start:endp) stateCollection3(:,start:endp) stateCollection4(:,start:endp) stateCollection5(:,start:endp)];
new_teacherCollection = [teacherCollection teacherCollection2 teacherCollection3 teacherCollection4 teacherCollection5];

outputWeights = feval(esn.methodWeightCompute, new_stateCollection, new_teacherCollection) ;
outputSequence = new_stateCollection * outputWeights(1,:)' ;

%outputWeights = feval(esn.methodWeightCompute, [stateCollection stateCollection2 stateCollection3 stateCollection4 stateCollection5], [teacherCollection teacherCollection2 teacherCollection3 teacherCollection4 teacherCollection5]) ;
%outputSequence = [stateCollection stateCollection2 stateCollection3 stateCollection4 stateCollection5] * outputWeights(1,:)' ;
nOutputPoints = length(outputSequence(:,1)) ;
outputSequence = feval(esn.outputActivationFunction, outputSequence);
outputSequence = outputSequence - repmat(esn.teacherShift',[nOutputPoints 1]) ;
predictedTrainOutput = outputSequence / diag(esn.teacherScaling) ;

% Print plots
figure(1);
title('Trained ESN');
hold on;
plot(trainOutputSequence(:,1), 'blue');
plot(trainInputSequence(:,1),'red');
plot(predictedTrainOutput(:,1),'black');

% extra figures for multiple outputs
% figure(2);
% title('cos');
% hold on;
% plot(trainOutputSequence(:,2), 'blue');
% plot(trainInputSequence(:,1),'red');
% plot(predictedTrainOutput(:,2),'black');
% 
% figure(3);
% title('sin');
% hold on;
% plot(trainOutputSequence(:,3), 'blue');
% plot(trainInputSequence(:,1),'red');
% plot(predictedTrainOutput(:,3),'black');
% 
% figure(4);
% title('sawtooth');
% hold on;
% plot(trainOutputSequence(:,4), 'blue');
% plot(trainInputSequence(:,1),'red');
% plot(predictedTrainOutput(:,4),'black');


%% Test on new test set
%predictedTestOutput = test_esn(testInputSequence,  trainedEsn, nForgetPoints) ; 

stateCollection = compute_statematrix(testInputSequence, testOutputSequence, esn, nForgetPoints) ;

stateCollection2 = compute_statematrix(testInputSequence, testOutputSequence, esn2, nForgetPoints) ;

stateCollection3 = compute_statematrix(testInputSequence, testOutputSequence, esn3, nForgetPoints) ;

stateCollection4 = compute_statematrix(testInputSequence, testOutputSequence, esn4, nForgetPoints) ;

stateCollection5 = compute_statematrix(testInputSequence, testOutputSequence, esn5, nForgetPoints) ;

new_stateCollection = [stateCollection(:,start:endp) stateCollection2(:,start:endp) stateCollection3(:,start:endp) stateCollection4(:,start:endp) stateCollection5(:,start:endp)];

outputSequence = new_stateCollection* outputWeights(1,:)' ;

%outputSequence = [stateCollection stateCollection2 stateCollection3 stateCollection4 stateCollection5]* outputWeights(1,:)' ;
nOutputPoints = length(outputSequence(:,1)) ;
outputSequence = feval(esn.outputActivationFunction, outputSequence);
outputSequence = outputSequence - repmat(esn.teacherShift',[nOutputPoints 1]) ;
predictedTestOutput = outputSequence / diag(esn.teacherScaling) ;

figure(2);
title('Test ESN');
hold on;
plot(trainOutputSequence(:,1), 'blue');
plot(trainInputSequence(:,1),'red');
plot(predictedTestOutput(:,1),'black');


%% Compute NRMSE error for both sets
 trainError = compute_NRMSE(predictedTrainOutput, trainOutputSequence); 
disp(sprintf('train NRMSE = %s', num2str(trainError)))
testError = compute_NRMSE(predictedTestOutput, testOutputSequence); 
disp(sprintf('test NRMSE = %s', num2str(testError)))
