function eeg_data_outlier = extracttrials_ecog(input_ECoG)


%% initialize variables

srate = 1000;                      % sampling rate of raw data in Hz
filterorder = 3;
filtercutoff = [3/500 30/500];
[f_b, f_a] = butter(filterorder,filtercutoff);
decimation = 10;                   % downsampling factor

%% preprocessing & filtering ECoG

f.data = input_ECoG ;
n_channels = size(f.data,1);
reference = 1:64;

% common baseline removing
ref = repmat(mean(f.data(reference,:),1),n_channels,1);
f.data = f.data - ref;
n_channels = size(f.data,1);

% bandpass filter the data (with a forward-backward filter)
for j = 1:n_channels
    f.data(j,:) = filtfilt(f_b,f_a,f.data(j,:));
end

% downsample the data (from 1000 Hz to 100 Hz)
ecog_data = f.data(:,1:decimation:end);

% outlier removing
w = windsor;
w = train(w,ecog_data,0.1);
eeg_data_outlier{1,1} = apply(w,ecog_data);



