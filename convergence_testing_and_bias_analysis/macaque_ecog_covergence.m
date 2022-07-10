
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

% A simple plot for check ECoG data
srate = 100;
scale_factor = std(Macaque_ECoG_outlier_removed(:))*3;
sjk_eeg_plot(Macaque_ECoG_outlier_removed , scale_factor  , srate)
xlabel('time (second)' ,'FontSize',12,'Interpreter' ,'latex' )
ylabel('ECoG channels' ,'FontSize',12,'Interpreter' ,'latex' )


%% LSIM modelling to check convergence for a 64-channel dataset

clear channels_observations
% to reduce running time we check only the first 10^5 samples
for d = 1:1
    for c =1:64
        channels_observations{c,d} = Macaque_ECoG_outlier_removed( (c-1)*1+1 : c*1,10000*(d-1)+1:10000*d);
    end
end

clear convergence_log Model_t Model_rep BIC_S

% set state numbers & Gaussian components
state_numbers = [2,3,4,5];
num_gmm_components = ones(1,length(state_numbers));

for s = 1: length(state_numbers)
    clc
    'lsim'
    s
    max_itration = 100;
    channel_num_states(1:size(channels_observations,1)) = state_numbers(s);
    num_gmm_component(1:size(channels_observations,1)) = num_gmm_components(s);

    extra.plot = 1;
    extra.check_convergence = 0;

    [pi_0_lsim , coupling_tetha_convex_comb , transition_matrices_convex_comb ,  lsim_gmm_para ,  AIC , log_likelihood , BIC] = ...
        em_lsim( channels_observations , channel_num_states , num_gmm_component , max_itration , extra);
    convergence_log(s,:) = log_likelihood ;

end

save([save_results_dir,'/ECoG_convergence.mat'],'convergence_log'); % save results

%% paper figure 3

clc
load([save_results_dir,'/ECoG_convergence.mat'])
close all

subplot_num = 'abcd';


figure('Position' ,  [200 200 750 450] ) % [left bottom width height]
plot( convergence_log' ,'LineWidth',1.5)
hold on
grid on
xlim([1,100])
legend({'$2~states$','$3~states$','$4~states$','$5~states$'},'FontSize',12,'Interpreter' ,'latex')
% title(['$\mathbf{',num2str(64),'~channels~ECoG}$'],'FontSize',11,'Interpreter' ,'latex')
set(gca, 'FontWeight','bold','FontSize',11);
xlabel('Iteration' ,'FontSize',13,'Interpreter' ,'latex' )
ylabel('Log-likelihood' ,'FontSize',13,'Interpreter' ,'latex' )




