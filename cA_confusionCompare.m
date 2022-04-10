% myKsDir = 'Z:\invivo_ephys\SharedEphys\FromSylvia\SS088_2018-01-30_K1\SS088_2018-01-30_K1_g0\SS088_2018-01-30_K1_g0_imec0\SS088_2018-01-30_K1_g0_t0_imec\imec_ks2\';
% myKsDir = 'D:\SharedEphysData\FromSyliva\SS088_2018-01-30_K2\';
myKsDir = 'D:\SharedEphysData\FerminoData\KilosortOut\Kilosort2_2021-03-13_180605\';


%% load results
metricFile = [myKsDir filesep 'cluster_group - copy.csv'];
T1 = readtable(metricFile);
    
metricFile = [myKsDir filesep 'noiseClassifier.csv'];
T2 = readtable(metricFile);

metricFile = [myKsDir filesep 'noiseModule.csv'];
T3 = readtable(metricFile);

%% confusion matrix - decoder
manualNoise = ismember(T1.group, 'noise');
decNoise = ismember(T2.group, 'noise');

decodeConfusion(1,1) = sum(~manualNoise & decNoise) / length(manualNoise); %false positive
decodeConfusion(2,1) = sum(manualNoise & decNoise) / length(manualNoise); %true positive
decodeConfusion(1,2) = sum(~manualNoise & ~decNoise) / length(manualNoise); %true negative
decodeConfusion(2,2) = sum(manualNoise & ~decNoise) / length(manualNoise); %false negative

%% confusion matrix - noise module
moduleNoise = ismember(T3.group, 'noise'); %only contains original clusters
redManualNoise = manualNoise(1 : length(moduleNoise)); %original clusters that were labeled as noise

moduleConfusion(1,1) = sum(~redManualNoise & moduleNoise) / length(redManualNoise); %false positive
moduleConfusion(2,1) = sum(redManualNoise & moduleNoise) / length(redManualNoise); %true positive
moduleConfusion(1,2) = sum(~redManualNoise & ~moduleNoise) / length(redManualNoise); %true negative
moduleConfusion(2,2) = sum(redManualNoise & ~moduleNoise) / length(redManualNoise); %false negative

%% feedback
disp('Noise decoder hit-rate:');
disp(sum(decNoise & manualNoise) / sum(manualNoise));

disp('Noise module hit-rate:');
disp(sum(moduleNoise & redManualNoise) / sum(redManualNoise));

disp('Noise decoder confusion-matrix:');
disp(decodeConfusion);

disp('Noise module confusion-matrix:');
disp(moduleConfusion);


