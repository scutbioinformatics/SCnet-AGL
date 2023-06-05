clc
clear
close all


load ORLdata28_23
%         alldata=data;
%         true_gt=gnd;
data=alldata;
gnd=true_gt';
clear alldata true_gt
% gnd=n*1;
% data=d*n;
[d,n]=size(data);
data=data-repmat(mean(data,2),[1,n]);
% data=data./max(max(data));


%% Initiation

dim_Reduce1=10;
% dim_Reduce2=floor(0.2*dim_Reduce1);
% dim_Reduce3=10;

% sita1=rand(d,1);
% sita1=sita1./sum(sita1);
%
% sita2=rand(dim_Reduce1,1);
% sita2=sita2./sum(sita2);
%
% sita3=rand(dim_Reduce2,1);
% sita3=sita3./sum(sita3);

sita1=(1./(d))*ones(d,1);
% sita2=(1./(dim_Reduce1))*ones(dim_Reduce1,1);
% sita3=(1./(dim_Reduce2))*ones(dim_Reduce2,1);
num_cluster=max(gnd);

alpha=1;

k=3;

lambda1=1e-5;
lambda2=1e-2;

iter_max=30;

options = [];
% options.NeighborMode = 'Supervised';
% options.gnd = gnd;
% options.WeightMode = 'HeatKernel';
options.NeighborMode = 'KNN';
options.k = k;
options.WeightMode = 'HeatKernel';
%       options.t = 1;
W = full(constructW(data',options));
D_mhalf =sum(W,2).^-.5;
step=0.3;
%% Main
obj=zeros(1,50);
for iter=1:50
    iter
    % forward
    [Y1]=forword_new(data,sita1,dim_Reduce1,lambda1,lambda2,W);
    Y1=real(Y1);
    %     [Y2]=forword_new(Y1,sita2,dim_Reduce2,k,lambda1,lambda2,W,D_mhalf);
    %
    %     [Y3]=forword_new(Y2,sita3,dim_Reduce3,k,lambda1,lambda2,W,D_mhalf);
    %     loss function
    if iter==1
        [~,C] =  kmeans(Y1',num_cluster,'MaxIter',1000,'Replicates',50);
    end
    [Y1,C,obj(iter)]=loss_function(Y1,num_cluster,alpha,C,step);
    Y1=real(Y1);
    % backward
    %     [sita3]=backword_layer(Y2,Y3,lambda2,iter_max);
    %     [sita2]=backword_layer(Y1,Y2,lambda2,iter_max);
    
    [sita1]=backword_layer(data,Y1,lambda2,iter_max);
    
    %         rng('default');
    %         temp_Y= tsne(Y3','Algorithm','exact','Standardize',true,'Perplexity',20);
    %         % Plot the result
    %         figure;
    %         hold on
    %         gscatter(temp_Y(:,1),temp_Y(:,2),gnd);
end


%%
class_num=max(gnd);
parfor ii=1:100
    ii
    %         kmeans(datafs',class_num,'MaxIter',1000,'Replicates',50);
    [idx,~] =  kmeans(Y1',num_cluster,'MaxIter',1000,'Replicates',50);
    [A_nmi_value(ii),A_ACC(ii),A_f(ii),A_p(ii),A_r(ii),A_Purity(ii),A_AR(ii),A_RI(ii),A_MI(ii),A_HI(ii),A_MIhat(ii)] = Cluster_Evaluation(idx,gnd);
    
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

save Result_MultiLayer1
