clc
clear
close all

load UmistData_25

true_gt=gnd';
alldata=data;
num_cluster=max(true_gt);
% num_class=max(true_gt);
num_X=size(alldata,2);

alldata=alldata-repmat(mean(alldata,2),[1,num_X]);
% for i=1:size(alldata,1)
%     alldata(i,:)=(alldata(i,:)-mean(alldata(i,:)));
%
% end

parfor ii=1:100
    
    Pre_Label=kmeans(alldata',num_cluster,'MaxIter',1000,'Replicates',50);
    %         [A_nmi_value(ii,pl),A_ACC(ii,pl),A_f(ii,pl),A_p(ii,pl),A_r(ii,pl),A_Purity(ii,pl),A_AR(ii,pl),A_RI(ii,pl),A_MI(ii,pl),A_HI(ii,pl),A_MIhat(ii,pl)]= Cluster_Evaluation(Pre_Label',true_gt);
    [A_nmi_value(ii),A_ACC(ii),A_f(ii),A_p(ii),A_r(ii),A_Purity(ii),A_AR(ii),A_RI(ii),A_MI(ii),A_HI(ii),A_MIhat(ii)] = Cluster_Evaluation(Pre_Label',true_gt);
end


MA_nmi_value=mean(A_nmi_value);
MA_ACC=mean(A_ACC);
MA_f=mean(A_f);
MA_p=mean(A_p);
MA_r=mean(A_r);
MA_Purity=mean(A_Purity);
MA_AR=mean(A_AR);
MA_RI=mean(A_RI);
MA_MI=mean(A_MI);
MA_HI=mean(A_HI);
MA_MIhat=mean(A_MIhat);

save Result_Umistbaseline
