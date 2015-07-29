%% An attempt to model the material by trying to produce similar behaviour 
% in a virtual network to an actual material. The materials compared to are
% pbma, LC and gold resistor. This particular script tries to evolve both
% internal weights and scaling parameters 

clear all;

load('pbma');
load('LC');
load('GResist');

rand('seed', 15); %15

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
%esn.internalWeights = esn.spectralRadius * esn.internalWeights_UnitSR;
%esn.inputWeights = rand(nInternalUnits, nInputUnits)*esn.inputScaling;
inputWeights =(2.0 * rand(nInternalUnits, nInputUnits)- 1.0);
%esn.inputWeights = inputWeights * esn.inputScaling;


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
mat_type = LC;
phys_out = [mat_type(1528:8195,:); mat_type(1528:8195,:); mat_type(1528:8195,:)];
OutputSequence = phys_out(1:Fs,:)';


% Split training set
train_fraction = 0.5 ; % use 50% in training and 50% in testing
[trainInputSequence,testInputSequence ] = ...
    split_train_test(InputSequence',train_fraction);
[trainOutputSequence,testOutputSequence ] = ...
    split_train_test(OutputSequence',train_fraction);

% evo param
popSize = 10;
numGens = 100000;
genotype = zeros(popSize,2+nInternalUnits.^2); %2 scaling parameters (spectral & input) and reservoir weights W
globalErr = inf;
mutRate = 10; %percent

%populate first gen randomly
genotype(1,1) = 4*rand-2;
genotype(1,2) = 4*rand-2;
genotype(1,3:end) = reshape(esn.internalWeights_UnitSR,1,length(genotype)-2);

for i = 2:popSize
    for j = 1:2
        genotype(i,j) = 4*rand-2;
    end
    % mutate weights
    genotype(i,3:end) = mutate(reshape(esn.internalWeights_UnitSR,1,length(genotype)-2),mutRate); %input: matrix to mutate, no. to mutate. output: new weight matrix
    %genotype(i,:) = mutate(reshape(esn.internalWeights_UnitSR,1,length(genotype)),mutRate);
end

figure;
%% Run network - simple network no feedback or scaling
for gen = 1:numGens
    %assign neuron type
    type = 'tanh';%'identity';
    trainError = zeros(popSize,1);
    
    for pop = 1:popSize
        % apply scaling
        esn.internalWeights = abs(reshape(genotype(pop,3:end), nInternalUnits, nInternalUnits)) * genotype(pop,1);
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
        
             if rand >= 0.5
                genotype(i,round(rand)+1) = 4*rand-2;
                genotype(i,round(rand)+1) = 4*rand-2;
            else
                % mutate weights
                mutMatrix = genotype(i,3:end);
                genotype(i,3:end) = mutate(mutMatrix,mutRate); %input: matrix to mutate, no. to mutate. output: new weight matrix
   
             end
    end
    plot(distHistory);
    drawnow
end

globalErr

%% show best resevoir config
esn.internalWeights = abs(reshape(best_genotype(3:end), nInternalUnits, nInternalUnits)) * best_genotype(1);
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

%states
figure;
plot(state);

figure;
plot(trainOutputSequence);

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



% Show physical fft
figure;
N = length(mat_type);
xdft = fft(mat_type);
xdft = xdft(1:N/2+1,:);
psdx = (1/(Fs*N)) * abs(xdft).^2;
psdx(2:end-1,:) = 2*psdx(2:end-1,:);
freq = 0:Fs/length(mat_type):Fs/2;
test = log10(psdx);

plot(freq,10*log10(psdx))
grid on
xlim([0 100]);
title('Periodogram Using FFT')
xlabel('Frequency (Hz)')
ylabel('Power/Frequency (dB/Hz)')
