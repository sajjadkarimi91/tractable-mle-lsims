
close all
clc
clear
addpath(genpath(pwd))

% common paths settings

mydir = pwd;
idcs = strfind(mydir,filesep);
% second parent folder contains the datasets
dataset_dir = [mydir(1:idcs(end-1)-1),'/DataSets/EPFL eeg dataset'];% EPFL dataset directory
save_results_dir = [mydir(1:idcs(end-1)-1),'/Results/',mydir(idcs(end-1)+1:end)]; % saving path

lsim_path = [mydir(1:idcs(end-1)-1),'/lsim karimi toolbox'];% enter the path of LSIM toolbox
addpath(lsim_path)

%% Analysis of Approximate Inference Bias
sub_numbers =[1,2,3,4,6,7,8,9];


for i= 1:length(sub_numbers)

    close all
    sub_numbers(i)

    for s=1:4
        load_path = [dataset_dir,'/subject',num2str(sub_numbers(i)),'/session',num2str(s)];
        save_path = [dataset_dir,'/subject',num2str(sub_numbers(i)),'/s',num2str(s)];
        eeg_data_outlier{s} =  extracttrials(load_path,save_path);
    end

    for s=1:4
        for ii = 1:6
            temp{1,(s-1)*6+ii} = eeg_data_outlier{s}{ii};
        end
    end

    if i==1

        srate = 64;
        %scale_factor = std(temp{1,1}(:))*3;
        scale_factor=20;
        sjk_eeg_plot(temp{1,1}(:,1:3*srate) , scale_factor  , srate)

        xlabel('time (second)' ,'FontSize',12,'Interpreter' ,'latex' )
        ylabel('EEG channels' ,'FontSize',12,'Interpreter' ,'latex' )

        pause(0.05)
        close all

    end

    %% select a subsets of EEG electrodes

    d = 1; %means only 4-channel subsets
    channels = [31 32 13 16]; % Fz, Cz, Pz, Oz

    C = length(channels);

    clear channels_observations num_gmm_component channel_num_states

    for t = 1:20
        for c = 1:C
            channels_observations{c,t} = temp{1,t}(channels(c),:);
        end
    end

    % set state numbers & Gaussian components
    num_gmm_component(1:C) = 1;
    channel_num_states(1:C) = 5;

    max_itration = 500;
    extra.plot=1;
    extra.check_convergence =0;
    extra.time_series = 0;

    [pi_0_lsim_approx , coupling_tetha_IM_approx , transition_matrices_IM_approx ,  gmm_para_lsim_approx , pi_0_lsim_exact, coupling_tetha_IM_exact, transition_matrices_IM_exact, gmm_para_lsim_exact,  log_likelyhood,log2_likelyhood, log_likelyhood_exact, log2_likelyhood_exact] =   ...
        em_lsim_exact( channels_observations , channel_num_states , num_gmm_component , max_itration , extra); % a function to comparison exact and approximate inference
    convergence_log{i,d} = log_likelyhood;
    convergence_log2{i,d} = log2_likelyhood;
    convergence_log_exact{i,d} = log_likelyhood_exact;
    convergence_log2_exact{i,d} = log2_likelyhood_exact;

    coupling_tetha{i,d} = coupling_tetha_IM_approx;
    transition_matrices{i,d} = transition_matrices_IM_approx;
    coupling_tetha_exact{i,d} = coupling_tetha_IM_exact;
    transition_matrices_exact{i,d} = transition_matrices_IM_exact;



end

if ~exist(save_results_dir)
    mkdir(save_results_dir)
end

save([save_results_dir,'/convergence_comparison_eeg.mat'] , 'transition_matrices_exact','transition_matrices','coupling_tetha_exact','coupling_tetha','convergence_log','convergence_log2','convergence_log_exact','convergence_log2_exact')

%% plot for first scenario

load([save_results_dir,'/convergence_comparison_eeg.mat'])
clc
close all

d = 1;
L = length(convergence_log_exact{1,d});

figure('Position' ,  [200 50 850 950] ) % [left bottom width height]
title(['$Exact~likelihood~curve~for~EM~alg~via~approximate~inference $'],'FontSize',15,'Interpreter' ,'latex')

for i = 1:8
    subplot(4,2,i)
    plot(2:L, convergence_log_exact{i,d}(2:end) ,'LineWidth',1.5)
    hold on
    grid on
    xlim([2,L])
    set(gca, 'FontWeight','bold','FontSize',9);
    xlabel('Iteration' ,'FontSize',12,'Interpreter' ,'latex' )
    ylabel('Log-likelihood' ,'FontSize',12,'Interpreter' ,'latex' )
end


sgtitle(['$Exact~likelihood~curve~for~EM~alg~via~approximate~inference~on~simulated~data$'],'FontSize',15,'Interpreter' ,'latex')

figure('Position' ,  [200 200 850 450] ) % [left bottom width height]
i = 1;
plot(2:L, convergence_log_exact{i,d}(2:end) ,'LineWidth',1.5)
hold on
plot(2:L, convergence_log{i,d}(2:end) ,'LineWidth',1.5)
grid on
xlim([2,L])
legend({'$Exact$','$Approx$'},'FontSize',11,'Interpreter' ,'latex')
title(['$Exact \& approx likelihood curves for EM alg via approximate inference $'],'FontSize',15,'Interpreter' ,'latex')
set(gca, 'FontWeight','bold','FontSize',9);
xlabel('Iteration' ,'FontSize',12,'Interpreter' ,'latex' )
ylabel('Log-likelihood' ,'FontSize',12,'Interpreter' ,'latex' )

%% plot for second scenario

load([save_results_dir,'/convergence_comparison_eeg.mat'])
clc
close all

d = 1;
L = length(convergence_log_exact{1,d});

num_channels = [4,8,16,32];

subplot_num = 'abcd';
marker_types = {'-+','-*','-x','-<','->','-o'};

figure('Position' ,  [200 50 850 950] ) % [left bottom width height]

for i = 1:8
    subplot(4,2,i)
    plot(1:L, convergence_log2_exact{i,d}(1:end) ,'LineWidth',1.5)
    hold on
    plot(1:L, convergence_log_exact{i,d}(1:end) ,'LineWidth',1.5)
    grid on
    xlim([1,L])
    if i==1
        legend({'$Exact~Inference$','$Approximate~Inference$'},'FontSize',12,'Interpreter' ,'latex')
    end
    set(gca, 'FontWeight','bold','FontSize',9);
    xlabel('Iteration' ,'FontSize',12,'Interpreter' ,'latex' )
    ylabel('Log-likelihood' ,'FontSize',12,'Interpreter' ,'latex' )

end

sgtitle(['$Exact~likelihood~curves~for~EM~alg~via~exact~\&~approximate~inference~on~EEG~data$'],'FontSize',15,'Interpreter' ,'latex')


for i = 1:8
    Log_lasts (2,i) = convergence_log_exact{i,d}(end);
    Log_lasts (1,i) = convergence_log2_exact{i,d}(end);
    Log_lasts (3,i) = Log_lasts (2,i) - Log_lasts (1,i);
end

err_coupling = [];
err_trans = [];
for i = 1:8
    err_coupling = [err_coupling;coupling_tetha{i,d}(:)- coupling_tetha_exact{i,d}(:)];
    temp1 = cell2mat(transition_matrices{i,d});
    temp2 = cell2mat(transition_matrices_exact{i,d});
    err_trans = [err_trans;temp1(:)- temp2(:)];
end

%% paper figure 8

figure('Position' ,  [200 200 450 250] ) % [left bottom width height]

for i = 2

    plot(1:L, convergence_log2_exact{i,d}(1:end) ,'LineWidth',1.5)
    hold on
    plot(1:L, convergence_log_exact{i,d}(1:end) ,'LineWidth',1.5)
    grid on
    xlim([1,L])
    legend({'$Exact~Inference$','$Approximate~Inference$'},'FontSize',12,'Interpreter' ,'latex')
    set(gca, 'FontWeight','bold','FontSize',9);
    xlabel('Iteration' ,'FontSize',12,'Interpreter' ,'latex' )
    ylabel('Log-likelihood' ,'FontSize',12,'Interpreter' ,'latex' )

end

