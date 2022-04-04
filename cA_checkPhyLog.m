function [isNoise, isMerge, noiseStr, mergeStr] = cA_checkPhyLog(cFolder)


%% get camlog file
logFile = dir(fullfile(cFolder, 'phy.log'));
cFile = fullfile(cFolder, logFile(1).name);

%% open file and run through all lines
fid = fopen(cFile);
isNoise = [];
isMerge = [];
noiseStr = {};
mergeStr = {};
while true
    tline = fgetl(fid); 
    
    % check for eof
    if not(ischar(tline))
        break
    end
    
    if contains(tline, 'Move') &&  contains(tline, 'to noise')
        val = textscan( tline(strfind(tline, 'Move'):end), '%s%f%s', 'delimiter', '['); %find number of noise cluster
        isNoise(end+1) = val{2};
        noiseStr{end+1} = tline;
    end
    
    if contains(tline, 'Merge clusters')
        startString = strfind(tline, 'Merge clusters ') + length('Merge clusters ');
        endString = strfind(tline, ' to ');
        val = textscan(tline(startString:endString), '%f%f%f%f%f%f%f', 'delimiter', ','); %find number of noise cluster
        val = [val{:}];
        isMerge(end + (1: length(val))) = val;
        mergeStr{end+1} = tline;
    end
end
fclose(fid);

%
isNoise = unique(isNoise);
isMerge = unique(isMerge);

