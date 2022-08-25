
close all
clc
clear
addpath(genpath(pwd))

% common path settings

mydir = pwd;
idcs = strfind(mydir,filesep);
dataset_dir = [mydir(1:idcs(end-1)-1),'/DataSets/EPFL eeg dataset']; % EPFL dataset directory
save_results_dir = [mydir(1:idcs(end-1)-1),'/Results/',mydir(idcs(end-1)+1:end)]; % saving path

lsim_path = [mydir(1:idcs(end-1)-1),'/lsim karimi toolbox'];% enter the path of LSIM toolbox
addpath(lsim_path)

%% EEG data modelling with LSIMs

% This script uses a part of loading & preproceesing codes from below paper with the following download link:
%  "An efficient P300-based brain-computer interface for disabled subjects"
% https://www.epfl.ch/labs/mmspg/research/page-58317-en-html/bci-2/bci_datasets/

sub_numbers =[1,2,3,4,6,7,8,9]; % 8 subjects in EPFL dataset


for i= 1:length(sub_numbers)

    close all
    sub_numbers(i)

    for s=1:4

        load_path = [dataset_dir,'/subject',num2str(sub_numbers(i)),'/session',num2str(s)];
        save_path = [dataset_dir,'/subject',num2str(sub_numbers(i)),'/s',num2str(s)];
        eeg_data_outlier_removed{s} =  extracttrials(load_path,save_path);

    end

    for s=1:4 % 4 sessions
        for r = 1:6 % each session has 6 recordings
            eeg_data_allsessions_records{1,(s-1)*6+r} = eeg_data_outlier_removed{s}{r};
        end
    end

    if i==1
        % A simple plot for check EEG data
        srate = 64;
        scale_factor=20;
        sjk_eeg_plot(eeg_data_allsessions_records{1,1}(:,1:3*srate) , scale_factor  , srate)
        xlabel('time (second)' ,'FontSize',12,'Interpreter' ,'latex' )
        ylabel('EEG channels' ,'FontSize',12,'Interpreter' ,'latex' )
        pause(0.05)
        close all

    end

    %% select a subsets of EEG electrodes

    for d = 1:4

        if d==1
            channels = [31 32 13 16]; % Fz, Cz, Pz, Oz
        elseif d==2
            channels = [31 32 13 16 11 12 19 20]; % Fz, Cz, Pz, Oz, P7, P3, P4, P8
        elseif d==3
            channels = [31 32 13 16 11 12 19 20 15 17 8 23 5 26 9 22];  % Fz, Cz, Pz, Oz, P7, P3, P4, P8, O1, O2, C3, C4, FC1, FC2, CP1, CP2
        else
            channels = 1:32; % All electrodes
        end

        % prepare data for LSIM
        C = length(channels);
        clear channels_observations num_gmm_component channel_num_states

        for t = 1:24
            for c = 1:C
                channels_observations{c,t} = eeg_data_allsessions_records{1,t}(channels(c),:);
            end
        end

        max_itration = 100;
        extra.plot = 1;
        extra.check_convergence = 0;
        extra.time_series = 0;

        % set state numbers & Gaussian components
        num_gmm_component(1:C) = 2;
        channel_num_states(1:C) = 4;

        % train the model
        [pi_0_lsim , coupling_tetha_convex_comb , transition_matrices_convex_comb ,  lsim_gmm_para ,  AIC , log_likelyhood , BIC] = ...
            em_lsim( channels_observations , channel_num_states , num_gmm_component , max_itration , extra);

        convergence_log{i,d} = log_likelyhood;

    end

end

if ~exist(save_results_dir)
    mkdir(save_results_dir)
end

save([save_results_dir,'/convergence_log.mat'] , 'convergence_log')


%% plot

sub_numbers =[1,2,3,4,6,7,8,9];
load([save_results_dir,'/convergence_log.mat'])
clc

close all

subplot_num = 'abcd';
marker_types = {'-+','-*','-x','-<','->','-o'};

figure('Position' ,  [200 200 850 450] ) % [left bottom width height]

num_channels = [4,8,16,32];

for d = 1:4

    subplot(2,2,d)

    for i = 1:8

        plot( convergence_log{i,d} ,'LineWidth',1.5)
        hold on

    end

    grid on
    xlim([1,100])

    if d==1
        legend({'$\#1$','$\#2$','$\#3$','$\#4$','$\#6$','$\#7$','$\#8$','$\#9$'},'FontSize',11,'Interpreter' ,'latex')
    end

    title(['$\mathbf{',subplot_num(d),'.~',num2str(num_channels(d)),'~channels}$'],'FontSize',15,'Interpreter' ,'latex')
    set(gca, 'FontWeight','bold','FontSize',9);
    xlabel('Iteration' ,'FontSize',12,'Interpreter' ,'latex' )
    ylabel('Log-likelihood' ,'FontSize',12,'Interpreter' ,'latex' )



end

%% paper figure 2

sub_numbers =[1,2,3,4,6,7,8,9];
load([save_results_dir,'/convergence_log.mat'])
clc

subplot_num = 'abab';
marker_types = {'-+','-*','-x','-<','->','-o'};

figure('Position' ,  [200 200 650 450] ) % [left bottom width height]

num_channels = [4,8,16,32];

for d = 3:4

    subplot(2,1,d-2)

    for i = 1:8
        plot( convergence_log{i,d} ,'LineWidth',1.5)
        hold on
    end

    grid on
    xlim([1,100])

    if d==3
        legend({'$\#1$','$\#2$','$\#3$','$\#4$','$\#6$','$\#7$','$\#8$','$\#9$'},'FontSize',11,'Interpreter' ,'latex')
    end

    title(['$\mathbf{',subplot_num(d),'.~',num2str(num_channels(d)),'~channels}$'],'FontSize',15,'Interpreter' ,'latex')
    set(gca, 'FontWeight','bold','FontSize',9);
    xlabel('Iteration' ,'FontSize',12,'Interpreter' ,'latex' )
    ylabel('Log-likelihood' ,'FontSize',12,'Interpreter' ,'latex' )

end







