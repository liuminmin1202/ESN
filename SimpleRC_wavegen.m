%% Very simple reservoir network with no scaling or feedback
%cord =[1,1; 1,2; 1,3; 1,4; 2,1; 2,2; 2,3; 2,4; 3,1; 3,2; 3,3; 3,4; 4,1; 4,2; 4,3; 4,4;];
clear all;

rand('seed', 15);

nInputUnits = 1; nInternalUnits = 16; nOutputUnits = 4; 
nForgetPoints = 200; % discard the first 100 points

% Scaling - input scaling and shift appear to affect the system the most
esn.spectralRadius = 1; %0.3
esn.inputScaling = 1; %0.3
esn.inputShift = 0; %-0.3

%define weights
connectivity = min([10/nInternalUnits 1]);
esn.internalWeights_UnitSR = generate_internal_weights(nInternalUnits, ...
                                                connectivity);   

% if wnat to load weights
% load('internalWeights');
% esn.internalWeights_UnitSR = weights;

% using no negative weights                                            
esn.internalWeights = esn.spectralRadius * abs(esn.internalWeights_UnitSR);
esn.inputWeights =abs(2.0 * rand(nInternalUnits, nInputUnits)- 1.0)*esn.inputScaling;


%% Assign input data and collect target output 
T = 10*(1/10);
Fs = 16666; %10x the freq being used
dt = 1/Fs;
t = 0:dt:T-dt;
%amplitude
A=1; %when weights are abs, anything above one produces bad results (probably saturating)

% Define input sequence
InputSequence(1,:)= A*sin(2*pi*10*t);

% Desired output
OutputSequence(1,:) = A*sawtooth(2*pi*10*t);
OutputSequence(2,:) = A*sin(2*pi*20*t);
OutputSequence(3,:) = A*square(2*pi*10*t);
OutputSequence(4,:) = A*cos(2*pi*10*t);
% Multiple outputs

% Split training set
train_fraction = 0.5 ; % use 50% in training and 50% in testing
[trainInputSequence,testInputSequence ] = ...
    split_train_test(InputSequence',train_fraction);
[trainOutputSequence,testOutputSequence ] = ...
    split_train_test(OutputSequence',train_fraction);

%% Run network - simple network no feedback or scaling
%assign neuron type
type = 'tanh';%'identity';

%Collect states
inputSequence = trainInputSequence;
state = zeros(length(inputSequence),nInternalUnits);
for i = 2:length(inputSequence)
    state(i,:) = feval(type,((esn.internalWeights*state(i-1,:)')+(esn.inputWeights*(inputSequence(i)+esn.inputShift))));
end

%add input as last state
state(:,nInternalUnits+1) = inputSequence;

%trim states to get rid of initial transient
states = state(nForgetPoints+1:end,:);

%show states
figure;
plot(states(:,1:16));

%calculate pseudoinverse to get output weights (batch-mode training process)
outputWeights = (pinv(states)*trainOutputSequence(nForgetPoints+1:end,:))';

% trained output
outputSequence = states * outputWeights';
trainError = compute_NRMSE(outputSequence, testOutputSequence)

%% Print plots
figure;
title('Trained ESN- saw');
hold on;
plot(trainOutputSequence(nForgetPoints+1:end,1), 'blue');
plot(trainInputSequence(nForgetPoints+1:end,1),'red');
plot(outputSequence(:,1),'black');

% extra figures for multiple outputs
figure;
title('2*sin');
hold on;
plot(trainOutputSequence(nForgetPoints+1:end,2), 'blue');
plot(trainInputSequence(nForgetPoints+1:end,1),'red');
plot(outputSequence(:,2),'black');

figure;
title('square');
hold on;
plot(trainOutputSequence(nForgetPoints+1:end,3), 'blue');
plot(trainInputSequence(nForgetPoints+1:end,1),'red');
plot(outputSequence(:,3),'black');

% 
figure;
title('cos');
hold on;
plot(trainOutputSequence(nForgetPoints+1:end,4), 'blue');
plot(trainInputSequence(nForgetPoints+1:end,1),'red');
plot(outputSequence(:,4),'black');

% Display Fast-fourier transform
figure;
temp_fft = abs(fft(states));
ft = temp_fft(1:length(states)/2,:);
f = Fs*(0:length(states)/2-1)/length(states);
plot(f,ft);
%grid on
xlim([0 100]);
title('Frequency response - FFT')
xlabel('Frequency (Hz)')
ylabel('|Y(f)|')

% power spectral density via fft
figure;
N = length(states);
xdft = fft(states);
xdft = xdft(1:N/2+1,:);
psdx = (1/(Fs*N)) * abs(xdft).^2;
psdx(2:end-1,:) = 2*psdx(2:end-1,:);
freq = 0:Fs/length(states):Fs/2;
test = log10(psdx);

plot(freq,10*log10(psdx))
grid on
xlim([0 100]);
title('Periodogram Using FFT')
xlabel('Frequency (Hz)')
ylabel('Power/Frequency (dB/Hz)')

