function syncMetric = pC_getSyncMetric(myKsDir, syncSpikesThresh, showPlot)
% usage: syncMetric = pC_getSyncMetric(myKsDir, syncSpikesThresh)
% function to detect synchronous spikes as a signature of noise clusters.
% Saves results to the 'metrics.csv' file if it exists. Otherwise, it
% creates a new file called 'syncSpike.csv'.
% myKsDir is the path to the kilosort output. syncSpikesThresh determines
% the amount of spikes that will be used to count as a 'sync event'.

% myKsDir = 'F:\Neuropixel_TestData\pipeline_analysis\rescaled_glbCAR\clean_data_CAR\37379_20201231_g0_t0_imec\imec_ks2';
myKsDir = 'Y:\invivo_ephys\Neuropixels\RD10_2129_20210112\RD10_2129_20210112_g0\RD10_2129_20210112_g0_imec0\RD10_2129_20210112_g0_t0_imec0\imec0_ks2';
%input variables
if ~exist('syncSpikesThresh', 'var') || isempty(syncSpikesThresh)
    syncSpikesThresh = [2 4 8 16]; %nr of synchronous spikes that are used for the metric. Default is 2, 4, 6, 8 and 16.
end


if ~exist('showPlot', 'var') || isempty(showPlot)
    showPlot = true ; %flag to show distribution for different noise metrics
end

metricFileName = 'metrics.csv'; %name of metrics file. Will add output to this file if it exists.

% load spike data
params.excludeNoise = false; %make sure to considered all clusters
sp = loadKSdir(myKsDir, params);  

%% get number of sync events events
syncEvents = (diff([0; sp.st])  == 0 | flipud(diff(flipud([sp.st; 0])))  == 0); %this will find spike times that match each other
syncCnt = diff([0; syncEvents]);
syncOn = find(syncCnt == 1);
syncOff = find(syncCnt == -1);
syncOff = syncOff(1:length(syncOn)); %make sure this matches in length
syncDiff = syncOff - syncOn; %number of synchronous spikes per event

cntSyncEvents = zeros(length(syncEvents), 1, 'single');
for x = unique(syncDiff')
    cIdx = syncDiff == x; %times with correct nr of sync spikes
    cIdx = syncOn(cIdx) + (0:x-1); %find onsets and add nr of sync spikes to the index
    cntSyncEvents(cIdx(:)) = x;
end

%% compute metric for different nr of sync events
clustIDs = unique(sp.clu); %get cluster IDs
syncMetric = zeros(length(clustIDs), length(syncSpikesThresh));
for x = 1: length(syncSpikesThresh)
    for iClust = 1 : length(clustIDs)
        syncMetric(iClust,x) = sum(sp.clu == clustIDs(iClust) & cntSyncEvents >= syncSpikesThresh(x)) / sum(sp.clu == clustIDs(iClust));
    end
end

%% check for csv file and add if possible
metricFile = [myKsDir filesep metricFileName];
if exist(metricFile, 'file')
    T = readtable(metricFile);

else
    metricFile = [myKsDir filesep 'syncSpike.csv'];
    Var1 = (0:max(clustIDs))';
    cluster_id = Var1;

    clear T
    T = table(Var1, cluster_id);
end


nClustIDs = T.cluster_id(:); %cluster IDs in csv file

%fill in results to table
for x = 1 : length(syncSpikesThresh)
    nSyncMetric = ones(length(nClustIDs), 1);
    for iClust = 1 : length(clustIDs)
        nSyncMetric(nClustIDs == clustIDs(iClust)) = syncMetric(iClust,x);
    end

    T.(['syncSpike_' num2str(syncSpikesThresh(x))]) = nSyncMetric;
    % check if this should be plotted
    if showPlot
        if x == 1
            h = figure('renderer','painters');
        end
        figure(h);
        subplot(1, length(syncSpikesThresh), x);
        histogram(nSyncMetric,0:0.01:1); %this will show the distribution of the syncmetric, in steps of 1%
        axis square; title(['syncSpike_' num2str(syncSpikesThresh(x))]);
    end
end


writetable(T, metricFile);
end 