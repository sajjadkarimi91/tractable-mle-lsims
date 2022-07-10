clc
clear
close all

addpath(genpath(pwd))

% common paths settings

mydir = pwd;
idcs = strfind(mydir,filesep);
save_results_dir = [mydir(1:idcs(end-1)-1),'/Results/',mydir(idcs(end-1)+1:end)]; % saving path

lsim_path = [mydir(1:idcs(end-1)-1),'/lsim karimi toolbox'];% enter the path of LSIM toolbox
addpath(lsim_path)

%% This demo generates one chmm model observation, then compare our approximate inference with the exact inference

%C is the number of channels in CHMM

max_repeat = 8;
max_C = 4; 
rng(0);
T = 1000; % T is the number of time samples

for C = 4:max_C % only simulate 4-channel cases
                    
    for rp = 1:max_repeat
        clc
        C
        rp
        close all
        channel_num_states(1:C) = 4;        
        channel_dim_observ(1:C) =  1;
        num_gmm_component(1:C) = 1; 

        clear transition_chmm pi_0_chmm chmm_gmm_para        
        for zee = 1:C
                        
            %pi_0 is the initial state probabilities for the channel zee            
            temp_var = rand( channel_num_states(zee) , 1);
            temp_var = temp_var / sum(temp_var);
            pi_0_chmm{zee,1} = temp_var;

            %transition probabilities            
            temp = abs(randn( channel_num_states(zee) , prod(channel_num_states)));
            temp = temp ./ repmat( sum(temp) , channel_num_states(zee) , 1) ;
            transition_chmm{zee,1} = temp';
            
            
            % Gaussian mixture model initialization for each state in channel zee
            for c=1:channel_num_states(zee)                
                temp_var = rand(num_gmm_component(zee) , 1);
                temp_var = temp_var / sum(temp_var);
                chmm_gmm_para{zee,1}.gmm_para(c).P = temp_var;                
                for k=1:num_gmm_component(zee)                    
                    chmm_gmm_para{zee,1}.gmm_para(c).mu(k).x =  1*c+randn(channel_dim_observ(zee) , 1);
                    chmm_gmm_para{zee,1}.gmm_para(c).sigma(k).x = 1+2*rand(channel_dim_observ(zee) , 1);                    
                end                
            end            
        end
 
        % generating equivalent HMM parameters to perform exact inference
        [ pi_0_ehmm , coupling_tetha_ehmm ,  transition_ehmm  ,ehmm_gmm_para  ] = chmm_cartesian_product( pi_0_chmm ,  transition_chmm  , chmm_gmm_para  );
       
        %generating observations time-series
        [ channels_observations , channel_hidden_states ] = generate_chmm_time_series( T , channel_num_states , channel_dim_observ , chmm_gmm_para , transition_chmm , pi_0_chmm );

        % training with approximate inference with the exact inference
        max_itration = 1000;
        extra.plot=1;
        extra.check_convergence =0;

        [pi_0_lsim_approx , coupling_tetha_IM_approx , transition_matrices_IM_approx ,  gmm_para_lsim_approx , pi_0_lsim_exact, coupling_tetha_IM_exact, transition_matrices_IM_exact, gmm_para_lsim_exact,  log_likelyhood,log2_likelyhood, log_likelyhood_exact, log2_likelyhood_exact] =  ...
            em_lsim_exact( channels_observations , channel_num_states , num_gmm_component , max_itration , extra);
        convergence_log{rp,C} = log_likelyhood;
        convergence_log2{rp,C} = log2_likelyhood;
        convergence_log_exact{rp,C} = log_likelyhood_exact;
        convergence_log2_exact{rp,C} = log2_likelyhood_exact;
        
        coupling_tetha{rp,C} = coupling_tetha_IM_approx;
        transition_matrices{rp,C} = transition_matrices_IM_approx;
        coupling_tetha_exact{rp,C} = coupling_tetha_IM_exact;
        transition_matrices_exact{rp,C} = transition_matrices_IM_exact;
        
    end        
end


if ~exist(save_results_dir)
    mkdir(save_results_dir)
end

save([save_results_dir,'/convergence_comparison_chmm4.mat'] , 'transition_matrices_exact','transition_matrices','coupling_tetha_exact','coupling_tetha','convergence_log','convergence_log2','convergence_log_exact','convergence_log2_exact')

%% plot for first scenario

load([save_results_dir,'/convergence_comparison_chmm4.mat'])
clc
close all

d = 4;
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

load([save_results_dir,'/convergence_comparison_chmm.mat'])
clc
close all

d = 4;
L = length(convergence_log_exact{1,d});


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

sgtitle(['$Exact~likelihood~curves~for~EM~alg~via~exact~\&~approximate~inference~on~simulated~data$'],'FontSize',15,'Interpreter' ,'latex')

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

for i = 8
    
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

