function [Y,A2,temp_deltaL]=forword_new(data,sita,dim_Reduce,lambda1,lambda2,W)
L=diag(sum(W,2))-W;
temp_sitadata=(diag(sita)*data);
A=temp_sitadata*temp_sitadata'+lambda2*eye(size(data,1));
% A_inv2=inv(A);
% A=(A'+A)/2;
% [U,E,V]=svd(A);
% A_inv2=V*(E^(-1))*U';
temp_deltaL=eye(size(data,2))-(temp_sitadata'/A)*temp_sitadata;
A2=eye(size(data,2))+lambda1*L-(temp_sitadata'/A)*temp_sitadata;

[Y, eigvalue] = eig(A2);
eigvalue = diag(eigvalue);

[junk, index] = sort(eigvalue);
eigvalue = eigvalue(index);
Y = Y(:,index);
if dim_Reduce< length(eigvalue)
    Y = Y(:, 1:dim_Reduce);
    eigvalue = eigvalue(1:dim_Reduce);
end
Y=Y';