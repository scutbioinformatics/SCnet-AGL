clc
clear
close all


load GTdata32

data=alldata;
gnd=true_gt';
clear alldata true_gt

[d,n]=size(data);
data=data-repmat(mean(data,2),[1,n]);



%% Initiation

dim_Reduce1=100;
dim_Reduce2=10;


sita1=(1./(d))*ones(d,1);
sita2=(1./(dim_Reduce1))*ones(dim_Reduce1,1);

num_cluster=max(gnd);

alpha=1;

k=3;

lambda1_L1=1e0;
lambda2_L1=1e3;
lambda1_L2=1e-2;
lambda2_L2=1e-4;

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
    [Y1]=forword_new(data,sita1,dim_Reduce1,lambda1_L1,lambda2_L1,W);
    Y1=real(Y1);
    Y1=FeaNorm_ljy(Y1);
    [Y2]=forword_new(Y1,sita2,dim_Reduce2,lambda1_L2,lambda2_L2,W);
    Y2=real(Y2);
    Y2=FeaNorm_ljy(Y2);

    %     loss function
    if iter==1
        [~,C] =  kmeans(Y2',num_cluster,'MaxIter',1000,'Replicates',50);
    end
    [Y2,C,obj(iter)]=loss_function(Y2,num_cluster,alpha,C,step);
    Y2=real(Y2);
    Y2=FeaNorm_ljy(Y2);
    
    % backward

    [sita2]=backword_layer(Y1,Y2,lambda2_L2,iter_max);
    
    [sita1]=backword_layer(data,Y1,lambda2_L1,iter_max);
    

end


%%
class_num=max(gnd);
for ii=1:50
    ii

    [idx,C] =  kmeans(Y2',class_num,'MaxIter',1000,'Replicates',50);
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


