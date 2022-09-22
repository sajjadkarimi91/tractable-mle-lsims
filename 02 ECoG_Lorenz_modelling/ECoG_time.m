
close all
clc
clear
addpath(genpath(pwd))

% common paths settings

mydir = pwd;
idcs = strfind(mydir,filesep);
dataset_dir = [mydir(1:idcs(end-1)-1),'/DataSets/20100802S1_Epidural-ECoG+Food-Tracking_B_Kentaro+Shimoda_mat_ECoG64-Motion6'];% Macaque ECoG dataset directory
save_results_dir = [mydir(1:idcs(end-1)-1),'/Results/',mydir(idcs(end-1)+1:end)];% saving path

lsim_path = [mydir(1:idcs(end-1)-1),'/chmm-lsim-karimi-toolbox'];% enter the path of LSIM toolbox
addpath(lsim_path)

%% data loading & preprocessing

for ch = 1:64
    temp = load([dataset_dir,'/ECoG_ch',num2str(ch),'.mat']);
    temp_cell = struct2cell(temp);
    if ch == 1
        Macaque_ECoG_channels = zeros(64,length(temp_cell{1}));
    end
    Macaque_ECoG_channels(ch,:) = temp_cell{1};
end

Macaque_ECoG_outlier_removed = extracttrials_ecog(Macaque_ECoG_channels);
Macaque_ECoG_outlier_removed = Macaque_ECoG_outlier_removed{1};

% computing cross-correlation matrix for plot
for i = 1:64
    for j = 1:64
        temp = sjk_autocrosscorrelation_missing( Macaque_ECoG_outlier_removed(i,:) , Macaque_ECoG_outlier_removed(j,:) , 1);
        Cxy(i,j) = temp(end);
    end
end

%% HMM & LSIM modelling for Macaque ECoG dataset

clear channels_observations

% due to have grid search on LSIMs with about 100 states per channel there is a time-consuming runing
% To reduce running time, we consider the first 10^5 samples
for d = 1:10
    for c =1:4 % every 16-electrods is considered as a one channel in LSIM
        channels_observations{c,d} = Macaque_ECoG_outlier_removed( (c-1)*16+1 : c*16,1000*(d-1)+1:1000*d);
    end
end


max_itration = 100;
%% training 4-channel LSIMs with differents number of states

clear Log_rep hmm_models model_rep BIC_hmm

state_numbers_lsim = [2,3,4];
state_numbers_all = state_numbers_lsim;
num_gmm_component_all = 1*ones(1,length(state_numbers_all));


C = size(channels_observations,1);
extra.plot = 0;
extra.check_convergence=0;


clc
disp('LSIMs')
for s = 1: length(state_numbers_all)
    disp(s)

    channel_num_states  = ones(1,C)*state_numbers_all(s);
    num_gmm_component  = ones(1,C)*num_gmm_component_all(s);

    tic
    em_lsim( channels_observations , channel_num_states , num_gmm_component , max_itration , extra);
    toc
end



%% training 4-channel CHMMs with differents number of states

clear Log_rep hmm_models model_rep BIC_hmm

state_numbers_chmm = 2:4;
state_numbers_all = state_numbers_chmm;
num_gmm_component_all = ones(1,length(state_numbers_all));

C = size(channels_observations,1);

extra.plot = 0;
extra.check_convergence=0;

disp('CHMMs')
for s = 1: length(state_numbers_all)
    disp(s)

    channel_num_states  = ones(1,C)*state_numbers_all(s);
    num_gmm_component  = ones(1,C)*num_gmm_component_all(s);
    tic
    em_chmm( channels_observations , channel_num_states , num_gmm_component , max_itration , extra);
    toc

end


