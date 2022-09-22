
close all
clc
clear
addpath(genpath(pwd))

% common paths settings
mydir = pwd;
idcs = strfind(mydir,filesep);
% second parent folder contains the datasets
save_results_dir = [mydir(1:idcs(end-1)-1),'/Results/',mydir(idcs(end-1)+1:end)]; % saving path

lsim_path = [mydir(1:idcs(end-1)-1),'/chmm-lsim-karimi-toolbox'];% enter the path of LSIM toolbox
addpath(genpath(lsim_path))

%% Simulate dataset


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


six_channel_lorenz_timeseries = [Lorenz_para_1(20:5:end,:), Lorenz_para_2(20:5:end,:)]';

% % figure
% % % plot3(Lorenz_para_1(:,1),Lorenz_para_1(:,2),Lorenz_para_1(:,3)) %'plot3' is the command to make 3D plot
% % % figure
% % % plot(Lorenz_para_1)

% % figure
% % plot3(Lorenz_para_2(:,1),Lorenz_para_2(:,2),Lorenz_para_2(:,3)) %'plot3' is the command to make 3D plot
% % figure
% % plot(Lorenz_para_2)


%% training 6-channel LSIMs with differents number of states

clear channels_observations


for c=1:size(six_channel_lorenz_timeseries,1)
    channels_observations{c,1}=six_channel_lorenz_timeseries( (c-1)+1 : c,:);
end
max_itration = 100;
max_rep = 2;

clear Log Model_t Model_rep BIC_hmm

state_numbers_lsim =  [2,3,4];
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

%% training 6-channel CHMMs with differents number of states

state_numbers_chmm = [2:4];
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




