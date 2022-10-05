close all
clc
clear
addpath(genpath(pwd))

mydir = pwd;
idcs = strfind(mydir,filesep);

addpath([mydir(1:idcs(end-1)-1),'/lsim karimi toolbox'])

feature_root_path = 'D:\PHD codes\DataSets\BED\Features\Verification/';
feature_names = {'MFCC.mat','ARRC.mat','SPEC.mat'};

stimuli_names = {'image','cognitive','rest_closed','rest_open','vepc3','vepc5','vepc7','vepc10',...
    'vep3','vep5','vep7','vep10'};

session_train = 1;
session_test = [2,3];


%%


for f=1:length(feature_names)

    path_this = [feature_root_path,feature_names{f}];
    load(path_this);

    for s=1:length(stimuli_names)
        clear Y this_data
        this_data = cell(1);
        c=1;
        for k=1:length(DATA)
            if strcmp(DATA(k).stimuli,stimuli_names{s})
                temp = DATA(k).data{1,1};%(f,t,ch)
                Y(c,:) = [DATA(k).y,DATA(k).session];
                for ch=1:size(temp,3)
                    this_data{ch,c} = squeeze(temp(:,:,ch));
                end
                c=c+1;
            end
        end

        subjects_all = unique(Y(:,1));
        num_subj = length(subjects_all);
        Y_org = Y;
        for k=1:num_subj
            ind_k = Y_org(:,1)==subjects_all(k);
            Y(ind_k,1)=k;
        end

        %% train lsim model
        CV_number = num_subj;
        clear lsim_train
        for i=1:num_subj
            ind_sub = Y(:,1)==i & Y(:,2)==session_train;
            lsim_train{1,i} = this_data(:,ind_sub);
        end

        C = size(this_data,1);
        max_r = 1;
        max_itration = 100;
        extra.plot = 0;
        extra.check_convergence=0;
        extra.sigma_diag = 1;

        state_numbers_grid = [2,3,4];
        %         num_gmm_component_grid = [1*ones(1,length(state_numbers_grid)),2*ones(1,length(state_numbers_grid)),3*ones(1,length(state_numbers_grid)),4*ones(1,length(state_numbers_grid)),5*ones(1,length(state_numbers_grid))];
        %         state_numbers_grid = [state_numbers_grid,state_numbers_grid,state_numbers_grid,state_numbers_grid,state_numbers_grid];

        num_gmm_component_grid = [1*ones(1,length(state_numbers_grid)),2*ones(1,length(state_numbers_grid)),3*ones(1,length(state_numbers_grid))];
        state_numbers_grid = [state_numbers_grid,state_numbers_grid,state_numbers_grid];
        counter = 0;
        clear coupling_tetha_all
        for repeat_num = 1:max_r

            for i = 1:CV_number
                close all
                clc
                counter = counter+1;
                disp(round(counter*100/(CV_number*max_r)))
                if size(lsim_train{i},2)>5
                    parfor ss = 1:length(state_numbers_grid)

                        channel_num_states = ones(1,C)*state_numbers_grid(ss);
                        num_gmm_component = ones(1,C)*num_gmm_component_grid(ss);

                        [pi_0_lsim , coupling_tetha_convex_comb , transition_matrices_convex_comb ,  lsim_gmm_para ,  AIC , log_likelihood , BIC ,pi_steady] = ...
                            em_lsim( lsim_train{i} , channel_num_states , num_gmm_component , max_itration , extra);

                        lsim_gmm_para_all{ss,i,repeat_num} =  lsim_gmm_para;
                        transitions_matrices_all{ss,i,repeat_num} = transition_matrices_convex_comb;
                        coupling_tetha_all{ss,i,repeat_num} = coupling_tetha_convex_comb;
                        pi_0_all{ss,i,repeat_num} = pi_0_lsim ;
                        AIC_all(ss,i,repeat_num) = AIC(end);
                        log_likelihood_all{ss,i,repeat_num} =log_likelihood;
                        BIC_all(ss,i,repeat_num) = BIC(end);
                    end
                    %                 save(['lsim_',feature_names{f},'_s_',num2str(s),'.mat'],'lsim_gmm_para_all','transitions_matrices_all','coupling_tetha_all','pi_0_all','AIC_all','log_likelihood_all','BIC_all')
                end
            end

            %             save(['lsim_',feature_names{f}(1:end-4),'_s_',num2str(s),'.mat'],'lsim_gmm_para_all','transitions_matrices_all','coupling_tetha_all','pi_0_all','AIC_all','log_likelihood_all','BIC_all')
        end

        %% test lsim models

        CV_number = num_subj;

        for sess_num = 1:length(session_test)
            true_labels_all=[];
            P_O_versys_all=[];
            AUC_all = nan(num_subj,1);
            for i=1:min(num_subj,min(size(coupling_tetha_all,2)))
                ind_sub = Y(:,2)==session_test(sess_num);
                clear lsim_test P_O_versys
                lsim_test = this_data(:,ind_sub);
                if ~isempty(coupling_tetha_all{1,i})
                    [~,ind_min] = min(AIC_all(:,i));
                    % ind_min = 1;
                    lsim_gmm_para = lsim_gmm_para_all{ind_min,i} ;
                    transition_matrices_convex_comb = transitions_matrices_all{ind_min,i} ;
                    coupling_tetha_convex_comb = coupling_tetha_all{ind_min,i};
                    pi_0_lsim = pi_0_all{ind_min,i} ;

                    for k = 1:size(lsim_test,2)
                        P_O_model  = forward_backward_lsim( pi_0_lsim , coupling_tetha_convex_comb  , transition_matrices_convex_comb ,  lsim_gmm_para , lsim_test(:,k) );
                        P_O_versys(k,1) = P_O_model/size(lsim_test{1,k},2);
                    end

                    true_labels = zeros(size(P_O_versys));
                    this_subjs = Y(Y(:,2)==session_test(sess_num),1);
                    true_labels(this_subjs==i)=1;

                    try
                        [X_auc,Y_auc,T,AUC] = perfcurve(true_labels,P_O_versys,1);
                        AUC_all(i) = AUC;
                        true_labels_all = [true_labels_all;true_labels];
                        P_O_versys_all = [P_O_versys_all;P_O_versys];
                    end

                end

            end
            auc_avg(f,s,sess_num) =mean(AUC_all,'omitnan');
            [X_auc,Y_auc,T,AUC] = perfcurve(true_labels_all,P_O_versys_all,1);
            auc_avg_all(f,s,sess_num) = AUC;
        end

    end
end

