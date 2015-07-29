%% Very simple reservoir network with no scaling or feedback
%cord =[1,1; 1,2; 1,3; 1,4; 2,1; 2,2; 2,3; 2,4; 3,1; 3,2; 3,3; 3,4; 4,1; 4,2; 4,3; 4,4;];
clear all;

load('pbma');
load('LC');
load('GResist');

rand('seed', 42); %15

nInputUnits = 1; nInternalUnits = 15; nOutputUnits = 1; 
nForgetPoints = 200; % discard the first 100 points

% Scaling - input scaling and shift appear to affect the system the most
esn.spectralRadius = 1; %0.3
esn.inputScaling = 1; %0.3
esn.inputShift = 0; %-0.3

%define weights
connectivity = min([10/nInternalUnits 1]);
esn.internalWeights_UnitSR = generate_internal_weights(nInternalUnits, ...
                                                connectivity);   

%load('internalWeights');
%esn.internalWeights_UnitSR = weights;

% using no negative weights                                            
esn.internalWeights = esn.spectralRadius * esn.internalWeights_UnitSR;
%esn.inputWeights = rand(nInternalUnits, nInputUnits)*esn.inputScaling; 
inputWeights =(2.0 * rand(nInternalUnits, nInputUnits)- 1.0);
esn.inputWeights = inputWeights * esn.inputScaling;


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
% OutputSequence(1,:) = A*sawtooth(2*pi*10*t);
% OutputSequence(2,:) = A*sin(2*pi*20*t);
% OutputSequence(3,:) = A*square(2*pi*10*t);
% OutputSequence(4,:) = A*cos(2*pi*10*t);

% LCs = [LC(1528:8195,:); LC(1528:8195,:); LC(1528:8195,:)];
% OutputSequence = LCs(1:Fs,:)';

resistors = [LC(1528:8195,:); LC(1528:8195,:); LC(1528:8195,:)];
OutputSequence = resistors(1:Fs,:)';
% Multiple outputs

% Split training set
train_fraction = 0.5 ; % use 50% in training and 50% in testing
[trainInputSequence,testInputSequence ] = ...
    split_train_test(InputSequence',train_fraction);
[trainOutputSequence,testOutputSequence ] = ...
    split_train_test(OutputSequence',train_fraction);

% evo param
popSize = 10;
numGens = 350;
genotype = zeros(10,2);
globalErr = inf;

%populate first gen randomly
for i = 1:popSize
    for j = 1:2
        genotype(i,j) = 4*rand-2;
    end
end

figure;
%% Run network - simple network no feedback or scaling
for gen = 1:numGens
%assign neuron type
type = 'tanh';%'identity';
trainError = zeros(popSize,1);

for pop = 1:popSize
esn.internalWeights = genotype(pop,1) * abs(esn.internalWeights_UnitSR);
esn.inputWeights = abs(inputWeights) * genotype(pop,2);

%Collect states
inputSequence = trainInputSequence;
state = zeros(length(inputSequence),nInternalUnits);
for i = 2:length(inputSequence)
    state(i,:) = feval(type,((esn.internalWeights*state(i-1,:)')+(esn.inputWeights*(inputSequence(i)))));%+esn.inputShift))));
end

%add input as last state
%state(:,nInternalUnits+1) = inputSequence;

%trim states to get rid of initial transient
%states = state(nForgetPoints+1:end,:);
%states = pbma(nForgetPoints+1:Fs/2,:)+esn.inputShift;

%calculate pseudoinverse to get output weights (batch-mode training process)
%outputWeights = (pinv(states)*trainOutputSequence(nForgetPoints+1:end,:))';

% trained output
%outputSequence = states * outputWeights';

trainError(pop,:) = sum(compute_NRMSE(state, trainOutputSequence));
% trainError = compute_NRMSE(outputSequence, testOutputSequence);
end

 [minErr,index] = min(trainError);
 distHistory(gen) = minErr;
 %assign new global best
 if minErr < globalErr
     best_genotype = genotype(index,:);
     best_gen = gen;
     globalErr = minErr;
    % global_best = listESN(index);
 end

 genotype(1,:) = best_genotype;
 
 for i = 2:popSize
     genotype(i,:) = best_genotype;
     genotype(i,round(rand)+1) = 4*rand-2;
     genotype(i,round(rand)+1) = 4*rand-2;
 end

 plot(distHistory);
 drawnow
end

globalErr

%% show best resevoir config
esn.internalWeights = best_genotype(1) * abs(esn.internalWeights_UnitSR);
esn.inputWeights = abs(inputWeights) * best_genotype(2);

%Collect states
inputSequence = trainInputSequence;
state = zeros(length(inputSequence),nInternalUnits);
for i = 2:length(inputSequence)
    state(i,:) = feval(type,((esn.internalWeights*state(i-1,:)')+(esn.inputWeights*(inputSequence(i)))));%+esn.inputShift))));
end

% %add input as last state
% state(:,nInternalUnits+1) = inputSequence;
% %trim states to get rid of initial transient
% states = state(nForgetPoints+1:end,:);
% %calculate pseudoinverse to get output weights (batch-mode training process)
% outputWeights = (pinv(states)*trainOutputSequence(nForgetPoints+1:end,:))';
% % trained output
% outputSequence = states * outputWeights';
% trainError = compute_NRMSE(outputSequence, trainOutputSequence)
outputSequence = state;

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
temp_fft = abs(fft(state));
ft = temp_fft(1:length(state)/2,:);
f = Fs*(0:length(state)/2-1)/length(state);
plot(f,ft);
%grid on
xlim([0 100]);
title('Frequency response - FFT')
xlabel('Frequency (Hz)')
ylabel('|Y(f)|')

% power spectral density via fft
figure;
N = length(state);
xdft = fft(state);
xdft = xdft(1:N/2+1,:);
psdx = (1/(Fs*N)) * abs(xdft).^2;
psdx(2:end-1,:) = 2*psdx(2:end-1,:);
freq = 0:Fs/length(state):Fs/2;
test = log10(psdx);

plot(freq,10*log10(psdx))
grid on
xlim([0 100]);
title('Periodogram Using FFT')
xlabel('Frequency (Hz)')
ylabel('Power/Frequency (dB/Hz)')

%

%
figure;
N = length(pbma);
xdft = fft(pbma);
xdft = xdft(1:N/2+1,:);
psdx = (1/(Fs*N)) * abs(xdft).^2;
psdx(2:end-1,:) = 2*psdx(2:end-1,:);
freq = 0:Fs/length(pbma):Fs/2;
test = log10(psdx);

plot(freq,10*log10(psdx))
grid on
xlim([0 100]);
title('Periodogram Using FFT')
xlabel('Frequency (Hz)')
ylabel('Power/Frequency (dB/Hz)')
% 
% figure;
% N = length(resistor);
% xdft = fft(resistor);
% xdft = xdft(1:N/2+1,:);
% psdx = (1/(Fs*N)) * abs(xdft).^2;
% psdx(2:end-1,:) = 2*psdx(2:end-1,:);
% freq = 0:Fs/length(resistor):Fs/2;
% test = log10(psdx);
% 
% plot(freq,10*log10(psdx))
% grid on
% xlim([0 100]);
% title('Periodogram Using FFT')
% xlabel('Frequency (Hz)')
% ylabel('Power/Frequency (dB/Hz)')
% 
% figure;
% N = length(LC);
% xdft = fft(LC);
% xdft = xdft(1:N/2+1,:);
% psdx = (1/(Fs*N)) * abs(xdft).^2;
% psdx(2:end-1,:) = 2*psdx(2:end-1,:);
% freq = 0:Fs/length(LC):Fs/2;
% test = log10(psdx);
% 
% plot(freq,10*log10(psdx))
% grid on
% xlim([0 100]);
% title('Periodogram Using FFT')
% xlabel('Frequency (Hz)')
% ylabel('Power/Frequency (dB/Hz)')