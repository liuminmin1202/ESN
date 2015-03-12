
clear all; 

load seed_list.mat;
rand('seed', seed_list(1));

%% Create reservoir
%scale inputs and teacher attributes
nInputUnits = 1; nInternalUnits = 100; nOutputUnits = 1; 
numElectrodes = 2; 
nForgetPoints = 100; % discard the first 100 points
sequenceLength = 1000;
inputScale = zeros(nInputUnits,1);
inputShift = zeros(nInputUnits,1);

for i = 1:nInputUnits
    inputScale(i,:) = 0.7206; 
    inputShift(i,:) = 0.4697;% necessary to correlate input and prediction/target
end

teacherScaling = zeros(nOutputUnits,1); teacherShift = zeros(nOutputUnits,1);

for i = 1:nOutputUnits
    teacherScaling(i,:) = 0.3;
    teacherShift(i,:) = -0.2;
end

%create reservoir with correct scaling
esn = generate_esn(nInputUnits, nInternalUnits, nOutputUnits, ...
    'spectralRadius',0.8948,'inputScaling',inputScale,'inputShift',inputShift, ...
    'teacherScaling',teacherScaling,'teacherShift',teacherShift,'feedbackScaling', 0, ...
    'type', 'plain_esn');


%% Assign input data and collect target output 
%[trainInputSequence, trainOutputSequence] = change_inputValue(inputValue, Ytarg, nInputUnits, nOutputUnits, sequenceLength);
T = 10*(1/10);
Fs = 1000;
dt = 1/Fs;
t = 0:dt:T-dt;
%amplitude
A=2; %between 0-15
trainInputSequence(1,:)=A*sin(2*pi*10*t);
% configuration voltages
%trainInputSequence(2,:)= 0;%A*cos(2*pi*20*t);
%trainInputSequence(3,:)= 0;%A*square(2*pi*20*t);



%output
trainOutputSequence(1,:) = A*sawtooth(2*pi*10*t);

% len = 1;
% for i =1:len
%     trainOutputSequence(((i-1)*len)+1,:) = A*square(2*pi*10*t);
%     trainOutputSequence(((i-1)*len)+2,:) = A*cos(2*pi*10*t);
%     trainOutputSequence(((i-1)*len)+3,:) = A*sin(2*pi*20*t);
%     trainOutputSequence(((i-1)*len)+4,:) = A*sawtooth(2*pi*10*t);
%     
%     %x = sawtooth(2*pi*10*t);
% end



train_fraction = 0.5 ; % use 50% in training and 50% in testing
[trainInputSequence,testInputSequence ] = ...
    split_train_test(trainInputSequence',train_fraction);
[trainOutputSequence,testOutputSequence ] = ...
    split_train_test(trainOutputSequence',train_fraction);

%% train reservoir
%scale the reservoir using the 'spectral radius' i.e. absolute eigenvector
esn.internalWeights = esn.spectralRadius * esn.internalWeights_UnitSR;
 
%collect states. 
[trainedEsn, stateCollection] = ...
    train_esn(trainInputSequence, trainOutputSequence , esn, nForgetPoints);

predictedTrainOutput = test_esn(trainInputSequence, trainedEsn, nForgetPoints);

figure(1);
title('Trained ESN');
hold on;
plot(trainOutputSequence(:,1), 'blue');
plot(trainInputSequence(:,1),'red');
plot(predictedTrainOutput(:,1),'black');

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

predictedTestOutput = test_esn(testInputSequence,  trainedEsn, nForgetPoints) ; 

figure(2);
title('Test ESN');
hold on;
plot(trainOutputSequence(:,1), 'blue');
plot(trainInputSequence(:,1),'red');
plot(predictedTestOutput(:,1),'black');

 trainError = compute_NRMSE(predictedTrainOutput, trainOutputSequence); 
disp(sprintf('train NRMSE = %s', num2str(trainError)))

%%%%compute NRMSE testing error
testError = compute_NRMSE(predictedTestOutput, testOutputSequence); 
disp(sprintf('test NRMSE = %s', num2str(testError)))