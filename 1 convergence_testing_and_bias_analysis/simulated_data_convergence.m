
clc
clear
close all

addpath(genpath(pwd))

mydir = pwd;
idcs = strfind(mydir,filesep);
lsim_path = [mydir(1:idcs(end-1)-1),'/lsim karimi toolbox'];% enter the path of LSIM toolbox
addpath(lsim_path)

%%

T = 200 ;   % T  number of time samples

for C = 2:5 %C is number of channels in CHMM


    clc
    C

    channel_num_states(1:C) = randi([2,6],1,C);
    while(sum(channel_num_states)>25)
        channel_num_states(1:C) = randi([2,6],1,C);
    end

    %observation dimension of each channel also initialized randomly between 1 to 5
    channel_dim_observ(1:C) = randi([1,5],1,C);
    num_gmm_component(1:C) = 1;

    clear transition_chmm transition_chmm_second

    for zee = 1:C

        %pi_0 is the initialized state probabilities for the channel zee
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


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % generating observations from CHMM parameters

    [ channels_observations , channel_hidden_states ] = generate_chmm_time_series( T , channel_num_states , channel_dim_observ , chmm_gmm_para , transition_chmm , pi_0_chmm );

    for c =1:C
        train_obs_chmm{c,1} = channels_observations{c};
    end


    %% Training CHMM from observations for model 1

    max_itration = 100;
    extra.plot=0;
    extra.check_convergence =0;

    num_gmm_component_temp(1:C) = 1;
    state_numbers = 2:7;
    clear log_likelyhood_lsim

    for s = 1:6
        channel_num_states_t =state_numbers(s)*ones(1,C);

        [pi_0_lsim_temp , coupling_tetha_convex_comb_temp , transition_matrices_convex_comb_temp ,  lsim_gmm_para_temp ,  AIC, log_likelyhood ] = ...
            em_lsim( train_obs_chmm , channel_num_states_t , num_gmm_component_temp , max_itration , extra);
        log_likelyhood_lsim(s,:) = log_likelyhood;
    end

    lsim_convergence_paths{1,c} = log_likelyhood_lsim;

end



%% plot

subplot_num = 'abcd';
figure('Position' ,  [200 200 850 450] ) % [left bottom width height]
num_channels = 2:5;

for d = 1:4

    subplot(2,2,d)
    plot( lsim_convergence_paths{1,d+1}' ,'LineWidth',1.5)
    hold on
    grid on
    xlim([1,100])

    if d==1
        legend({'$2~states$','$3~states$','$4~states$','$5~states$','$6~states$','$7~states$'},'FontSize',11,'Interpreter' ,'latex')
    end

    title(['$\mathbf{',subplot_num(d),'.~',num2str(num_channels(d)),'~channels}$'],'FontSize',15,'Interpreter' ,'latex')
    set(gca, 'FontWeight','bold','FontSize',9);
    xlabel('Iteration' ,'FontSize',12,'Interpreter' ,'latex' )
    ylabel('Log-likelihood' ,'FontSize',12,'Interpreter' ,'latex' )


end


%% paper figure 1

clc

subplot_num = 'abcd';
figure('Position' ,  [200 200 650 650] ) % [left bottom width height]
num_channels = [3,5];

for d = 1:2

    subplot(2,1,d)

    if d==1
        plot( lsim_convergence_paths{1,3}' ,'LineWidth',1.5)
        hold on
    else
        plot( lsim_convergence_paths{1,5}' ,'LineWidth',1.5)
        hold on
    end
    grid on
    xlim([1,100])

    if d==1
        legend({'$2~states$','$3~states$','$4~states$','$5~states$','$6~states$','$7~states$'},'FontSize',11,'Interpreter' ,'latex')
    end

    title(['$\mathbf{',subplot_num(d),'.~',num2str(num_channels(d)),'~channels}$'],'FontSize',15,'Interpreter' ,'latex')
    set(gca, 'FontWeight','bold','FontSize',9);
    xlabel('Iteration' ,'FontSize',12,'Interpreter' ,'latex' )
    ylabel('Log-likelihood' ,'FontSize',12,'Interpreter' ,'latex' )

end

