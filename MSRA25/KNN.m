clear
clc
close all

for pl=1:10
    pl
    eval(['load DataORL4_',num2str(pl)]);

%  options=[];
% options.ReducedDim=181;
% options.PCARatio=0.99;
% [eigvector,eigvalue] = PCA(train',options);
% train = eigvector'*train; 
% test = eigvector'*test; 
ACC_mrr(pl)= kNN_classifier(1,  train', train_gt',  test' ,test_gt');


end
ACC=mean(ACC_mrr);
save Result_ORL_KNN