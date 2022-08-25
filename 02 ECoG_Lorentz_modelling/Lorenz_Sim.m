
close all
clc
clear
addpath(genpath(pwd))

mydir = pwd;
idcs = strfind(mydir,filesep);
% second parent folder contains the datasets
dataset_dir = [mydir(1:idcs(end-1)-1),'/DataSets'];
results_dir = [mydir(1:idcs(end-1)-1),'/Results/',mydir(idcs(end-1)+1:end)];

addpath([mydir(1:idcs(end-1)-1),'/lsim karimi toolbox'])

%%


%Solution for the Lorenz equations in the time interval [0,100] with initial conditions [1,1,1].
sigma=10;
beta=8/3;
rho=28;
f = @(t,a) [-sigma*a(1) + sigma*a(2); rho*a(1) - a(2) - a(1)*a(3); -beta*a(3) + a(1)*a(2)];
%'f' is the set of differential equations and 'a' is an array containing values of x,y, and z variables.
%'t' is the time variable
[t1,Lorenz_para_1] = ode45(f,[0:0.01:200],[1 1 1]);%'ode45' uses adaptive Runge-Kutta method of 4th and 5th order to solve differential equations



%Solution for the Lorenz equations in the time interval [0,100] with initial conditions [1,1,1].
sigma=10;
beta=8/3;
rho=56;
f = @(t,a) [-sigma*a(1) + sigma*a(2); rho*a(1) - a(2) - a(1)*a(3); -beta*a(3) + a(1)*a(2)];
%'f' is the set of differential equations and 'a' is an array containing values of x,y, and z variables.
%'t' is the time variable
[t2,Lorenz_para_2] = ode45(f,[0:0.01:200],[1 0 0]);%'ode45' uses adaptive Runge-Kutta method of 4th and 5th order to solve differential equations


multichannel_timeseries_lorenz = [Lorenz_para_1(20:5:end,:), Lorenz_para_2(20:5:end,:)]';


% % % plot3(Lorenz_para_1(:,1),Lorenz_para_1(:,2),Lorenz_para_1(:,3)) %'plot3' is the command to make 3D plot
% % % figure
% % % plot(Lorenz_para_1)

% % figure
% % plot3(Lorenz_para_2(:,1),Lorenz_para_2(:,2),Lorenz_para_2(:,3)) %'plot3' is the command to make 3D plot
% % figure
% % plot(Lorenz_para_2)




%% hmm & lsim modelling

clear channels_observations


for c=1:size(multichannel_timeseries_lorenz,1)
    channels_observations{c,1}=multichannel_timeseries_lorenz( (c-1)+1 : c,:);
end



clear Log Model_t Model_rep BIC_S

state_numbers_all = [2,3,4,5,6,7,8:3:25];
num_gmm_component_all = [ones(1,length(state_numbers_all)),2*ones(1,length(state_numbers_all)),3*ones(1,length(state_numbers_all))];
state_numbers_all = [state_numbers_all,state_numbers_all,state_numbers_all];


for s_search = 1: length(state_numbers_all)
    clc
    'lsim'
    s_search
    max_itration = 200;
    channel_num_states(1:size(channels_observations,1)) = state_numbers_all(s_search);
    num_gmm_component(1:size(channels_observations,1)) = num_gmm_component_all(s_search);
    extra.plot = 0;
    extra.check_convergence=0;
    
    parfor replicate_number = 1:3
        
        try
            
            
            [pi_0_lsim , coupling_tetha_convex_comb , transition_matrices_convex_comb ,  lsim_gmm_para ,  AIC , log_likelihood , BIC ,pi_steady] = ...
                em_lsim( channels_observations , channel_num_states , num_gmm_component , max_itration , extra);
            
            Model_rep{replicate_number}.CHMM_GMM_Param = lsim_gmm_para;
            Model_rep{replicate_number}.Transitions_matrices = transition_matrices_convex_comb;
            Model_rep{replicate_number}.Coupling_Tetha = coupling_tetha_convex_comb;
            Model_rep{replicate_number}.PI_0 = pi_0_lsim;
            Log(replicate_number) = log_likelihood(end);
            BIC_Rep(replicate_number) =  BIC(end);
            AIC_Rep(replicate_number) =  AIC(end);
            
        catch
            Log(replicate_number)=-inf;
        end
        
    end
    
    [~,Index_max] = max(Log);
    Model_lsim{s_search}.CHMM_GMM_Param =    Model_rep{Index_max}.CHMM_GMM_Param ;
    Model_lsim{s_search}.Transitions_matrices = Model_rep{Index_max}.Transitions_matrices;
    Model_lsim{s_search}.Coupling_Tetha = Model_rep{Index_max}.Coupling_Tetha;
    Model_lsim{s_search}.PI_0 = Model_rep{Index_max}.PI_0 ;
    BIC_lsim_S(s_search) = BIC_Rep(Index_max);
    AIC_lsim_S(s_search) = AIC_Rep(Index_max);
    
end

[~,Index_min] = min(BIC_lsim_S);
Model{1}.CHMM_GMM_Param =    Model_lsim{Index_min}.CHMM_GMM_Param ;
Model{1}.Transitions_matrices = Model_lsim{Index_min}.Transitions_matrices;
Model{1}.Coupling_Tetha = Model_lsim{Index_min}.Coupling_Tetha;
Model{1}.PI_0 = Model_lsim{Index_min}.PI_0 ;


clear channels_observations channel_num_states  num_gmm_component
channels_observations{1,1} = multichannel_timeseries_lorenz;

for d = 1:1
    channels_observations{1,d} = multichannel_timeseries_lorenz;
end


state_numbers_all = [2,3,4,5,6,7:5:100];
num_gmm_component_all = [ones(1,length(state_numbers_all)),2*ones(1,length(state_numbers_all)),3*ones(1,length(state_numbers_all))];
state_numbers_all = [state_numbers_all,state_numbers_all,state_numbers_all];


for s_search = 1: length(state_numbers_all)
    
    clc
    'hmm'
    s_search
    max_itration = 200;
    channel_num_states(1:size(channels_observations,1)) = state_numbers_all(s_search);
    num_gmm_component(1:size(channels_observations,1)) = num_gmm_component_all(s_search);
    extra.plot=1;
    extra.check_convergence=0;
    
    parfor replicate_number = 1:3
        
        try
            
            [pi_0_lsim , coupling_tetha_convex_comb , transition_matrices_convex_comb ,  lsim_gmm_para ,  AIC , log_likelihood , BIC ,pi_steady] = ...
                em_lsim( channels_observations , channel_num_states , num_gmm_component , max_itration , extra);
            
            Model_rep{replicate_number}.CHMM_GMM_Param = lsim_gmm_para;
            Model_rep{replicate_number}.Transitions_matrices = transition_matrices_convex_comb;
            Model_rep{replicate_number}.Coupling_Tetha = coupling_tetha_convex_comb;
            Model_rep{replicate_number}.PI_0 = pi_0_lsim;
            Log(replicate_number) = log_likelihood(end);
            BIC_Rep(replicate_number) =  BIC(end);
            AIC_Rep(replicate_number) =  AIC(end);
            
        catch
            Log(replicate_number)=-inf;
        end
        
    end
    
    [~,Index_max] = max(Log);
    Model_t{s_search}.CHMM_GMM_Param =    Model_rep{Index_max}.CHMM_GMM_Param ;
    Model_t{s_search}.Transitions_matrices = Model_rep{Index_max}.Transitions_matrices;
    Model_t{s_search}.Coupling_Tetha = Model_rep{Index_max}.Coupling_Tetha;
    Model_t{s_search}.PI_0 = Model_rep{Index_max}.PI_0 ;
    BIC_S(s_search) = BIC_Rep(Index_max);
    AIC_S(s_search) = AIC_Rep(Index_max);
    
end

[~,Index_min] = min(BIC_S);
Model{2}.CHMM_GMM_Param =    Model_t{Index_min}.CHMM_GMM_Param ;
Model{2}.Transitions_matrices = Model_t{Index_min}.Transitions_matrices;
Model{2}.Coupling_Tetha = Model_t{Index_min}.Coupling_Tetha;
Model{2}.PI_0 = Model_t{Index_min}.PI_0 ;

mkdir(results_dir)
save([results_dir,'/Lorentz.mat'],'Model','BIC_S','BIC_lsim_S','AIC_S','AIC_lsim_S','Model_lsim','Model_t')


%%

load([results_dir,'/Lorentz.mat'])

close all

state_numbers_hmm = [2,3,4,5,6,7:5:100];
state_numbers_lsim = [2,3,4,5,6,7,8:3:25];

AIC_S = reshape(AIC_S,[],3);
AIC_lsim_S = reshape(AIC_lsim_S,[],3);
BIC_S = reshape(BIC_S,[],3);
BIC_lsim_S = reshape(BIC_lsim_S,[],3);

% AIC_S = AIC_S(:,1);
% AIC_lsim_S = AIC_lsim_S(:,1);
% BIC_S = BIC_S(:,1);
% BIC_lsim_S = BIC_lsim_S(:,1);

[AIC_S,ind_min] = min(AIC_S,[],2);
AIC_lsim_S = min(AIC_lsim_S,[],2);
BIC_S = min(BIC_S,[],2);
BIC_lsim_S = min(BIC_lsim_S,[],2);

subplot(2,1,1)
temp = AIC_S;
plot(state_numbers_hmm , temp,'LineWidth',1.5)
hold on

temp = AIC_lsim_S;
plot(state_numbers_lsim, temp,'LineWidth',1.5)

temp = AIC_S;
[~,ind_min] = min(temp);
plot(state_numbers_hmm(ind_min) , temp(ind_min),'bx','LineWidth',1.5)


temp = AIC_lsim_S;
[~,ind_min] = min(temp);
plot(state_numbers_lsim(ind_min) , temp(ind_min),'rx','LineWidth',1.5)

grid on
grid minor

set(gca, 'FontWeight','bold','FontSize',9,'XScale','log','Xtick', [2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80]);
xlabel('State number' ,'FontSize',12,'Interpreter' ,'latex' )
ylabel('AIC' ,'FontSize',12,'Interpreter' ,'latex' )

xlim([2,90])
legend('HMM','LSIM','Interpreter' ,'latex')

subplot(2,1,2)
temp = BIC_S;
plot(state_numbers_hmm,temp,'LineWidth',1.5)
hold on
[~,ind_min] = min(temp);
plot(state_numbers_hmm(ind_min) , temp(ind_min),'bx','LineWidth',1.5)

temp = BIC_lsim_S;
plot(state_numbers_lsim,temp,'LineWidth',1.5)
[~,ind_min] = min(temp);
plot(state_numbers_lsim(ind_min) , temp(ind_min),'rx','LineWidth',1.5)

grid on
grid minor

set(gca, 'FontWeight','bold','FontSize',9,'XScale','log','Xtick', [2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80]);
xlabel('State number' ,'FontSize',12,'Interpreter' ,'latex' )
ylabel('BIC' ,'FontSize',12,'Interpreter' ,'latex' )
xlim([2,90])
ylim([1.2*10^5,2.4*10^5])


figure
imagesc(Model{1}.Coupling_Tetha)
colorbar
set(gca, 'FontWeight','bold','FontSize',11);

