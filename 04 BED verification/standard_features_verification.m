close all
clc
clear
addpath(genpath(pwd))

mydir = pwd;
idcs = strfind(mydir,filesep);

addpath([mydir(1:idcs(end-1)-1),'/lsim karimi toolbox'])

feature_root_path = 'E:\share\S.karimi\PHD codes\DataSets\BED\Features\Verification/';
feature_names = {'MFCC.mat','ARRC.mat','SPEC.mat'};

stimuli_names = {'image','cognitive','rest_closed','rest_open','vepc3','vepc5','vepc7','vepc10',...
    'vep3','vep5','vep7','vep10'};

session_train = 1;
session_test = [1,2,3];


%%


for f=2:2

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
                    for cf = 1:size(temp,1)
                        this_data{ch}{cf,c} = squeeze(temp(cf,:,ch));
                    end
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
            for ch=1:size(temp,3)
                lsim_train{ch,i} = this_data{ch}(:,ind_sub);
            end
        end

        C = size(this_data{1},1);
        C_EEG = length(this_data);
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


        for i = 1:CV_number
            close all
            clc
            counter = counter+1;
            disp(round(counter*100/(CV_number)))
            if size(lsim_train{1,i},2)>5
                parfor ss = 1:length(state_numbers_grid)
                    for ch_eeg = 1:C_EEG
                        channel_num_states = ones(1,C)*state_numbers_grid(ss);
                        num_gmm_component = ones(1,C)*num_gmm_component_grid(ss);

                        [pi_0_lsim , coupling_tetha_convex_comb , transition_matrices_convex_comb ,  lsim_gmm_para ,  AIC , log_likelihood , BIC ,pi_steady] = ...
                            em_lsim( lsim_train{ch_eeg,i} , channel_num_states , num_gmm_component , max_itration , extra);

                        lsim_gmm_para_all{ss,i,ch_eeg} =  lsim_gmm_para;
                        transitions_matrices_all{ss,i,ch_eeg} = transition_matrices_convex_comb;
                        coupling_tetha_all{ss,i,ch_eeg} = coupling_tetha_convex_comb;
                        pi_0_all{ss,i,ch_eeg} = pi_0_lsim ;
                        AIC_all(ss,i,ch_eeg) = AIC(end);
                        log_likelihood_all{ss,i,ch_eeg} =log_likelihood;
                        BIC_all(ss,i,ch_eeg) = BIC(end);
                    end
                end
                %                 save(['lsim_',feature_names{f},'_s_',num2str(s),'.mat'],'lsim_gmm_para_all','transitions_matrices_all','coupling_tetha_all','pi_0_all','AIC_all','log_likelihood_all','BIC_all')

            end

            save(['lsim_',feature_names{f}(1:end-4),'_s_',num2str(s),'.mat'],'lsim_gmm_para_all','transitions_matrices_all','coupling_tetha_all','pi_0_all','AIC_all','log_likelihood_all','BIC_all')
        end

        %% test lsim models

        CV_number = num_subj;
        all_pobservs =cell(min(num_subj,min(size(coupling_tetha_all,2))),1 );
        all_truelabels =cell(min(num_subj,min(size(coupling_tetha_all,2))),1 );
        for sess_num = 1:length(session_test)

            AUC_all = nan(num_subj,1);
            AUC_chanels_all = nan(num_subj,1);
            AUC_all_best = nan(num_subj,1);
            AUC_chanels_all_best = nan(num_subj,1);

            for i=1:min(num_subj,min(size(coupling_tetha_all,2)))
                ind_sub = Y(:,2)==session_test(sess_num);
                clear lsim_test P_O_versys

                if ~isempty(coupling_tetha_all{1,i,1})
                    for ch_eeg = 1:C_EEG

                        lsim_test = this_data{ch_eeg}(:,ind_sub);
                        [~,ind_min] = min(squeeze(AIC_all(:,i,ch_eeg)));
                        % ind_min = 1;
                        lsim_gmm_para = lsim_gmm_para_all{ind_min,i,ch_eeg} ;
                        transition_matrices_convex_comb = transitions_matrices_all{ind_min,i,ch_eeg} ;
                        coupling_tetha_convex_comb = coupling_tetha_all{ind_min,i,ch_eeg};
                        pi_0_lsim = pi_0_all{ind_min,i,ch_eeg} ;

                        parfor k = 1:size(lsim_test,2)
                            P_O_model  = forward_backward_lsim( pi_0_lsim , coupling_tetha_convex_comb  , transition_matrices_convex_comb ,  lsim_gmm_para , lsim_test(:,k) );
                            P_O_versys(k,ch_eeg) = P_O_model/size(lsim_test{1,k},2);
                        end
                    end
                    true_labels = zeros(size(P_O_versys,1),1);
                    this_subjs = Y(Y(:,2)==session_test(sess_num),1);
                    true_labels(this_subjs==i)=1;
                    P_O_versys1 = sum(P_O_versys,2);
                    p_595 = prctile(P_O_versys,[5 95],'all');
                    thr_cands = linspace(p_595(1),p_595(2),20);
                    all_pobservs{i} = P_O_versys;
                    all_truelabels{i} = true_labels;

                    try

                        clear AUC_tmp
                        for thrC = 1:C_EEG
                            c_thr = 1;
                            for thr = thr_cands
                                [~,~,~,AUC_tmp(thrC,c_thr)] = perfcurve(true_labels,double(sum(P_O_versys>thr,2)>thrC),1);
                                c_thr=c_thr+1;
                            end
                        end
                        [AUC_chanels_all(i),I] = max(AUC_tmp(:));
                        if sess_num==1
                            [I1,I2] = ind2sub(size(AUC_tmp),I);
                            save_thr(i,1)= thr_cands(I2);
                            save_thr(i,2)=I1;
                        end
                        [X_auc,Y_auc,T,AUC] = perfcurve(true_labels,double(sum(P_O_versys>save_thr(i),2)>save_thr(i,2)),1);
                        AUC_all(i) = AUC;

                        clear AUC_tmp
                        c_thr = 1;
                        for thr = thr_cands
                            [~,~,~,AUC_tmp(c_thr)] = perfcurve(true_labels,sum(P_O_versys>thr,2),1);
                            c_thr=c_thr+1;
                        end

                        [AUC_chanels_all_best(i),I] = max(AUC_tmp(:));
                        if sess_num==1
                            save_thr_best(i,1)= thr_cands(I);
                        end

                        [X_auc,Y_auc,T,AUC] = perfcurve(true_labels, sum(P_O_versys>save_thr_best(i),2),1);
                        AUC_all_best(i) = AUC;

                    end
                end

            end
            estlabels{f,s,sess_num} = all_pobservs;
            truelabels{f,s,sess_num} = all_truelabels;
            auc_avg(f,s,sess_num) =mean(AUC_all,'omitnan');
            auc_avg_all(f,s,sess_num) = mean(AUC_chanels_all,'omitnan');
            auc_avg_best(f,s,sess_num) =mean(AUC_all_best,'omitnan');
            auc_avg_all_best(f,s,sess_num) = mean(AUC_chanels_all_best,'omitnan');
        end
        save(['results_hmm_',num2str(f),'.mat'],'auc_avg','auc_avg_all',"auc_avg_best","auc_avg_all_best",'estlabels',"truelabels")
    end
end

