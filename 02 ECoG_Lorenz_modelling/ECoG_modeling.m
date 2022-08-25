
close all
clc
clear
addpath(genpath(pwd))

% common paths settings

mydir = pwd;
idcs = strfind(mydir,filesep);
dataset_dir = [mydir(1:idcs(end-1)-1),'/DataSets/20100802S1_Epidural-ECoG+Food-Tracking_B_Kentaro+Shimoda_mat_ECoG64-Motion6'];% Macaque ECoG dataset directory
save_results_dir = [mydir(1:idcs(end-1)-1),'/Results/',mydir(idcs(end-1)+1:end)];% saving path

lsim_path = [mydir(1:idcs(end-1)-1),'/lsim karimi toolbox'];% enter the path of LSIM toolbox
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


%% training 4-channel LSIMs with differents number of states

clear Log_rep hmm_models model_rep BIC_hmm

state_numbers_all = [2,3,4,5,6,10:3:30,35:5:100];
num_gmm_component_all = [ones(1,length(state_numbers_all)),2*ones(1,length(state_numbers_all)),3*ones(1,length(state_numbers_all))];
state_numbers_all = [state_numbers_all,state_numbers_all,state_numbers_all];

C = size(channels_observations,1);

extra.plot = 0;
extra.check_convergence=0;

parfor s = 1: length(state_numbers_all)

    max_itration = 200;
    channel_num_states  = ones(1,C)*state_numbers_all(s);
    num_gmm_component  = ones(1,C)*num_gmm_component_all(s);

    BIC_rep = zeros(3,1);
    AIC_rep = zeros(3,1);
    Log_rep = zeros(3,1);
    model_rep = cell(3,1);

    for rp = 1:3

        try

            [pi_0_lsim , coupling_tetha_convex_comb , transition_matrices_convex_comb ,  lsim_gmm_para ,  AIC , log_likelihood , BIC ,pi_steady] = ...
                em_lsim( channels_observations , channel_num_states , num_gmm_component , max_itration , extra);

            model_rep{rp}.lsim_gmm_para = lsim_gmm_para;
            model_rep{rp}.transition_matrices = transition_matrices_convex_comb;
            model_rep{rp}.coupling_tetha = coupling_tetha_convex_comb;
            model_rep{rp}.pi_0 = pi_0_lsim;
            Log_rep(rp) = log_likelihood(end);
            BIC_rep(rp) =  BIC(end);
            AIC_rep(rp) =  AIC(end);

        catch
            Log_rep(rp)=-inf;
        end

    end

    [~,ind_best_rp] = max(Log_rep);
    lsim_models{s}.lsim_gmm_para =    model_rep{ind_best_rp}.lsim_gmm_para ;
    lsim_models{s}.transition_matrices = model_rep{ind_best_rp}.transition_matrices;
    lsim_models{s}.coupling_tetha = model_rep{ind_best_rp}.coupling_tetha;
    lsim_models{s}.pi_0 = model_rep{ind_best_rp}.pi_0 ;
    BIC_lsim(s) = BIC_rep(ind_best_rp);
    AIC_lsim(s) = AIC_rep(ind_best_rp);

end

[~,Index_min] = min(BIC_lsim);
best_lsim_bic{1}.lsim_gmm_para = lsim_models{Index_min}.lsim_gmm_para ;
best_lsim_bic{1}.transition_matrices = lsim_models{Index_min}.transition_matrices;
best_lsim_bic{1}.coupling_tetha = lsim_models{Index_min}.coupling_tetha;
best_lsim_bic{1}.pi_0 = lsim_models{Index_min}.pi_0 ;

%% training HMMs with differents number of states

clear channels_observations channel_num_states  num_gmm_component

for d = 1:10
    hmm_observations{1,d} = Macaque_ECoG_outlier_removed(:,1000*(d-1)+1:1000*d);
end

state_numbers_all = [2,3,4,5,6,7,10,15,20,25,30,40,50:5:100,105:8:250];
num_gmm_component_all = [ones(1,length(state_numbers_all)),2*ones(1,length(state_numbers_all)),3*ones(1,length(state_numbers_all))];
state_numbers_all = [state_numbers_all,state_numbers_all,state_numbers_all];

max_itration = 200;
extra.plot=0;
extra.check_convergence=0;

parfor s = 1: length(state_numbers_all)

    channel_num_states = state_numbers_all(s);
    num_gmm_component= num_gmm_component_all(s);
    BIC_rep = zeros(3,1);
    AIC_rep = zeros(3,1);
    Log_rep = zeros(3,1);
    model_rep = cell(3,1);

    for rp = 1:3

        try

            [pi_0_lsim , coupling_tetha_convex_comb , transition_matrices_convex_comb ,  lsim_gmm_para ,  AIC , log_likelihood , BIC ,pi_steady] = ...
                em_lsim( hmm_observations , channel_num_states , num_gmm_component , max_itration , extra);

            model_rep{rp}.lsim_gmm_para = lsim_gmm_para;
            model_rep{rp}.transition_matrices = transition_matrices_convex_comb;
            model_rep{rp}.coupling_tetha = coupling_tetha_convex_comb;
            model_rep{rp}.pi_0 = pi_0_lsim;
            Log_rep(rp) = log_likelihood(end);
            BIC_rep(rp) =  BIC(end);
            AIC_rep(rp) =  AIC(end);

        catch
            Log_rep(rp)=-inf;
        end

    end

    [~,ind_best_rp] = max(Log_rep);

    hmm_models{s}.lsim_gmm_para =    model_rep{ind_best_rp}.lsim_gmm_para ;
    hmm_models{s}.transition_matrices = model_rep{ind_best_rp}.transition_matrices;
    hmm_models{s}.coupling_tetha = model_rep{ind_best_rp}.coupling_tetha;
    hmm_models{s}.pi_0 = model_rep{ind_best_rp}.pi_0 ;
    BIC_hmm(s) = BIC_rep(ind_best_rp);
    AIC_hmm(s) = AIC_rep(ind_best_rp);

end

[~,Index_min] = min(BIC_hmm);
best_lsim_bic{2}.lsim_gmm_para =    hmm_models{Index_min}.lsim_gmm_para ;
best_lsim_bic{2}.transition_matrices = hmm_models{Index_min}.transition_matrices;
best_lsim_bic{2}.coupling_tetha = hmm_models{Index_min}.coupling_tetha;
best_lsim_bic{2}.pi_0 = hmm_models{Index_min}.pi_0 ;

mkdir(save_results_dir)
save([save_results_dir,'/Macaque_hmm_lsim.mat'],'BIC_lsim','BIC_hmm','AIC_lsim','AIC_hmm','lsim_models','hmm_models','Cxy','best_lsim_bic')

%%

load([save_results_dir,'/Macaque_hmm_lsim.mat'])
close all

state_numbers_hmm =  [2,3,4,5,6,7,10,15,20,25,30,40,50:5:100,105:8:250];
state_numbers_lsim =  [2,3,4,5,6,10:3:30,35:5:100];


AIC_hmm = reshape(AIC_hmm,[],3);
AIC_lsim = reshape(AIC_lsim,[],3);
BIC_hmm = reshape(BIC_hmm,[],3);
BIC_lsim = reshape(BIC_lsim,[],3);

% selecting the best Gaussian number for each state
AIC_hmm = min(AIC_hmm,[],2);
AIC_lsim = min(AIC_lsim,[],2);
BIC_hmm = min(BIC_hmm,[],2);
BIC_lsim = min(BIC_lsim,[],2);

subplot(2,1,1)
temp = AIC_hmm;
plot(state_numbers_hmm , temp,'LineWidth',1.5)
hold on

temp = AIC_lsim;
plot(state_numbers_lsim, temp,'LineWidth',1.5)

temp = AIC_hmm;
[~,ind_min] = min(temp);
plot(state_numbers_hmm(ind_min) , temp(ind_min),'bx','LineWidth',1.5)
temp = AIC_lsim;
[~,ind_min] = min(temp);
plot(state_numbers_lsim(ind_min) , temp(ind_min),'rx','LineWidth',1.5)

grid on
grid minor

set(gca, 'FontWeight','bold','FontSize',9,'XScale','log','Xtick', [2,3,4,5,6,7,8,9,10,20,30,40,60,80,100,150,200]);
xlabel('State number' ,'FontSize',12,'Interpreter' ,'latex' )
ylabel('AIC' ,'FontSize',12,'Interpreter' ,'latex' )

xlim([2,150])
legend('HMM','LSIM','Interpreter' ,'latex')

subplot(2,1,2)
temp = BIC_hmm;
plot(state_numbers_hmm,temp,'LineWidth',1.5)
hold on
[~,ind_min] = min(temp);
plot(state_numbers_hmm(ind_min) , temp(ind_min),'bx','LineWidth',1.5)

temp = BIC_lsim;
plot(state_numbers_lsim,temp,'LineWidth',1.5)
[~,ind_min] = min(temp);
plot(state_numbers_lsim(ind_min) , temp(ind_min),'rx','LineWidth',1.5)

grid on
grid minor

set(gca, 'FontWeight','bold','FontSize',9,'XScale','log','Xtick', [2,3,4,5,6,7,8,9,10,20,30,40,60,80,100,150,200]);
xlabel('State number' ,'FontSize',12,'Interpreter' ,'latex' )
ylabel('BIC' ,'FontSize',12,'Interpreter' ,'latex' )
xlim([2,150])
ylim([2.4*10^6,2.8*10^6])

%%
figure

subplot(2,1,1)
imagesc(Cxy)
colorbar
set(gca, 'FontWeight','bold','FontSize',11);
title('a. Cross-correlation matrix of ECoG','FontSize',11,'Interpreter' ,'latex')

subplot(2,1,2)
imagesc(best_lsim_bic{1}.coupling_tetha)
colorbar
set(gca, 'FontWeight','bold','FontSize',11);
title('b. Coupling weights of best model for ECoG based on BIC','FontSize',11,'Interpreter' ,'latex')


