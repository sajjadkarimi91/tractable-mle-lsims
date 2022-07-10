function eeg_data_outlier_removed = extracttrials(indir, outfile)
%
% extracttrials(indir, outfile)
%
% Extracts single trials from the files in *indir* and writes them to
% *outfile*.
%
% Example: extracttrials('subject1\session1','s1')

% Author: Ulrich Hoffmann - EPFL, 2006
% Copyright: Ulrich Hoffmann - EPFL

%% scan indir for input files
d = dir(indir);
filelist = {};
for i = 1:length(d)
    if(d(i).isdir == 0)
        filename = sprintf('%s\\%s',indir,d(i).name);
        filelist = cat(2,filelist,{filename});
    end
end
fprintf('found %i files ...\n',length(filelist));


%% initialize variables
runs = cell(1,length(filelist));
srate = 2048;                      % sampling rate of raw data in Hz
reference = [7 24]; %1:32;         % indices of channels used as reference
filterorder = 3;
filtercutoff = [1/1024 30/1024];
[f_b, f_a] = butter(filterorder,filtercutoff);
decimation = 32;                   % downsampling factor


%% loading, preprocessing & filtering EEG

for i = 1:length(filelist)

    % load data
    f = load(filelist{i});
    fprintf('processing %s\n',filelist{i});

    % rereference the data
    n_channels = size(f.data,1);
    ref = repmat(mean(f.data(reference,:),1),n_channels,1);
    f.data = f.data - ref;

    % drop the mastoid channels
    f.data = f.data(1:32,:);
    n_channels = size(f.data,1);

    % bandpass filter the data (with a forward-backward filter)
    for j = 1:n_channels
        f.data(j,:) = filtfilt(f_b,f_a,f.data(j,:));
    end

    % downsample the data (from 2048 Hz to 64 Hz)
    eeg_data = f.data(:,1:decimation:end);
    eeg_data(:,3001:end) =[]; % drop the first pne-minute due to high movement artifact

    % outlier removing
    w = windsor;
    w = train(w,eeg_data,0.1);
    eeg_data_outlier_removed{1,i} = apply(w,eeg_data);

end


%% save results in outfile
save(outfile,'runs');

