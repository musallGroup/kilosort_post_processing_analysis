function [syncMetric, spaceSyncMetric, farSyncMetric, nearSyncMetric] = pC_getSyncMetric(myKsDir, syncSpikesThresh, showPlot)
% usage: [syncMetric, spaceSyncMetric, farSyncMetric, nearSyncMetric] = pC_getSyncMetric(myKsDir, syncSpikesThresh, showPlot)
% function to detect synchronous spikes as a signature of noise clusters.
% Saves results to the 'metrics.csv' file if it exists. Otherwise, it
% creates a new file called 'syncSpike.csv'.
% myKsDir is the path to the kilosort output. syncSpikesThresh determines
% the amount of spikes that will be used to count as a 'sync event'.
%
% 'syncMetric' is the ratio of synchronous vs all spikes, 'spaceSyncMetric'
% is the average distance between synchronous channels, 'farSyncMetric' is
% syncMetric multiplied with channel distance to emphasize distributed sync
% events. 'nearSyncMetric' is syncMetric multiplied by 1-channel distance
% to emphasize close-by sync events.

% myKsDir = 'F:\Neuropixel_TestData\pipeline_analysis\rescaled_glbCAR\clean_data_CAR\37379_20201231_g0_t0_imec\imec_ks2';
% myKsDir = 'Y:\invivo_ephys\Neuropixels\RD10_2129_20210112\RD10_2129_20210112_g0\RD10_2129_20210112_g0_imec0\RD10_2129_20210112_g0_t0_imec0\imec0_ks2';

%input variables
if ~exist('syncSpikesThresh', 'var') || isempty(syncSpikesThresh)
    syncSpikesThresh = [2 4 8]; %nr of synchronous spikes that are used for the metric. Default is 2, 4, 6, 8 and 16.
end


if ~exist('showPlot', 'var') || isempty(showPlot)
    showPlot = true ; %flag to show distribution for different noise metrics
end

metricFileName = 'metrics.csv'; %name of metrics file. Will add output to this file if it exists.

% load spike data
params.excludeNoise = false; %make sure to considered all clusters
params.loadPCs = true;
sp = loadKSdir(myKsDir, params);

% get spatial distance for contacts in each sync events (based on the 'ksDriftmap' function) 
pcFeat = sp.pcFeat;
pcFeat = squeeze(pcFeat(:,1,:)); % take first PC only
pcFeat(pcFeat<0) = 0; % some entries are negative, but we don't really want to push the CoM away from there.
spikeFeatInd = sp.pcFeatInd(sp.spikeTemplates+1,:);
spikeFeatYcoords = sp.ycoords(spikeFeatInd+1); % 2D matrix of size #spikes x 12
spikeDepths = sum(spikeFeatYcoords.*pcFeat.^2,2)./sum(pcFeat.^2,2);
spikeDepths = spikeDepths / max(spikeDepths); %normalize to 1

%% get number of sync events
syncEvents = (diff([0; sp.st])  == 0 | flipud(diff(flipud([sp.st; 0])))  == 0); %this will find spike times that match each other
syncCnt = diff([0; syncEvents]);
syncOn = find(syncCnt == 1);
syncOff = find(syncCnt == -1);
syncOff = syncOff(1:length(syncOn)); %make sure this matches in length
syncDiff = syncOff - syncOn; %number of synchronous spikes per event

cntSyncEvents = zeros(length(syncEvents), 1, 'single'); %nr of sync spikes
spaceSyncEvents = zeros(length(syncEvents), 1, 'single'); %spatial spread of involved channels
for x = unique(syncDiff')
    cIdx = syncDiff == x; %times with correct nr of sync spikes
    cIdx = (syncOn(cIdx) + (0:x-1))'; %find onsets and add nr of sync spikes to the index
    cntSyncEvents(cIdx(:)) = x;
    
    spaceSpread = nanvar(spikeDepths(cIdx)); %range of depths for each sync event
    spaceSpread = repmat(spaceSpread, x, 1); %repeat to match size of index
    spaceSyncEvents(cIdx(:)) = spaceSpread;
end


%% compute metric for different nr of sync events
clustIDs = unique(sp.clu); %get cluster IDs
syncMetric = zeros(length(clustIDs), length(syncSpikesThresh));
spaceSyncMetric = zeros(length(clustIDs), length(syncSpikesThresh));
for x = 1: length(syncSpikesThresh)
    for iClust = 1 : length(clustIDs)
        
        cSpikes = sp.clu == clustIDs(iClust); %spikes from current cluster
        syncMetric(iClust,x) = sum(cSpikes & cntSyncEvents >= syncSpikesThresh(x)) / sum(cSpikes); % regular sync metric (ratio of sync vs all spikes)
        if syncMetric(iClust,x) > 0
            spaceSyncMetric(iClust,x) = nanmean(spaceSyncEvents(cSpikes & cntSyncEvents >= syncSpikesThresh(x)));  % spatially-corrected sync metric
        end
    end
end

% normalize between 0 and 1
spaceSyncMetric = spaceSyncMetric - min(spaceSyncMetric(:));
spaceSyncMetric = spaceSyncMetric ./ max(spaceSyncMetric(:));

farSyncMetric = syncMetric .* spaceSyncMetric;  % emphasizes sync events with distant channels
nearSyncMetric = syncMetric .* (1-spaceSyncMetric);  % emphasizes sync events with closeby channels
            
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
    nSpaceSyncMetric = ones(length(nClustIDs), 1);
    nFarSyncMetric = ones(length(nClustIDs), 1);
    nNearSyncMetric = ones(length(nClustIDs), 1);
    for iClust = 1 : length(clustIDs)
        nSyncMetric(nClustIDs == clustIDs(iClust)) = syncMetric(iClust,x);
        nSpaceSyncMetric(nClustIDs == clustIDs(iClust)) = spaceSyncMetric(iClust,x);
        nFarSyncMetric(nClustIDs == clustIDs(iClust)) = farSyncMetric(iClust,x);
        nNearSyncMetric(nClustIDs == clustIDs(iClust)) = nearSyncMetric(iClust,x);
    end
    
    T.(['syncSpike_' num2str(syncSpikesThresh(x))]) = nSyncMetric;
    T.(['syncSpace_' num2str(syncSpikesThresh(x))]) = nSpaceSyncMetric;
    T.(['farSyncSpike_' num2str(syncSpikesThresh(x))]) = nFarSyncMetric;
    T.(['nearSyncSpike_' num2str(syncSpikesThresh(x))]) = nNearSyncMetric;
    
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
