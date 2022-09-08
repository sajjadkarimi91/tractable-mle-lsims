addpath(genpath(pwd))
clear
close all
clc

mydir = pwd;
idcs = strfind(mydir,filesep);
% second parent folder contains the datasets
dataset_dir = [mydir(1:idcs(end-1)-1),'/DataSets'];
results_dir = [mydir(1:idcs(end-1)-1),'/Results/',mydir(idcs(end-1)+1:end)];
addpath([mydir(1:idcs(end-1)-1),'/chmm-lsim-karimi-toolbox'])

%%

test_number = 10;
max_repeat = 1000;


for C = [3,5] %C is number of channels in CHMM
    
    for T = 5:5:15    % T  number of time samples
        
        k1 = 0;
        for train_number = [10,15,20,25,30]
            
            k1=k1+1;
            
            clc
            C
            T
            train_number
            
            parfor repeat_num = 1:max_repeat

                [scores_lsim(repeat_num,:), scores_hmm(repeat_num,:), scores_svm(repeat_num,:), Label(repeat_num,:)] = sim_calassification(C,T,train_number, test_number);
                
            end
            
            
            Label = repmat(Label,max_repeat,1);
            ACC_lsim(C,T,train_number) = sum((scores_lsim(:)>0 &Label(:)>0.5)|(scores_lsim(:)<=0 & Label(:)<0.5))/(length(Label(:)));
            ACC_hmm(C,T,train_number) = sum((scores_hmm(:)>0 &Label(:)>0.5)|(scores_hmm(:)<=0 & Label(:)<0.5))/(length(Label(:)));
            ACC_svm(C,T,train_number) = sum((scores_svm(:)>0 &Label(:)>0.5)|(scores_svm(:)<=0 & Label(:)<0.5))/(length(Label(:)));
        
            close all
            mean(ACC_lsim(:)-ACC_svm(:))
            
            
            
            [~,~,~,AUC] = perfcurve(Label(:),scores_lsim(:),+1);
            AUC_chmm(C,T,train_number) = AUC;
            
            [~,~,~,AUC] = perfcurve(Label(:),scores_hmm(:),+1);
            AUC_hmm(C,T,train_number) = AUC;
            
            [~,~,~,AUC] = perfcurve(Label(:),scores_svm(:),+1);
            AUC_svm(C,T,train_number) = AUC;
            
            
            est_chmm{C,T,train_number} = scores_lsim;
            est_hmm{C,T,train_number} = scores_hmm;
            est_svm{C,T,train_number} = scores_svm;
            
            close all
            
        end
        
    end
end


save([results_dir,'/lsim_hmm_svm.mat'])


%% Plot

clc
load([results_dir,'/lsim_hmm_svm.mat'])


C = 2:6; %C is number of channels in CHMM
T = 5:5:15;    % T  number of time samples
train_number = [10,15,20,25,30];


AUC_chmm(C,T,train_number) ;
AUC_hmm(C,T,train_number) ;
AUC_svm(C,T,train_number) ;



