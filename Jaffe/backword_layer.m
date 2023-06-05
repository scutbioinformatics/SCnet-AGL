function [sita]=backword_layer(X,Y,lambda2,iter_max)

% X=d*n
% Y=k*n
O=eye(size(X,1));
e=1e-3;
obj=[];
for iter=1:iter_max
W=(X*X'+lambda2*O)\(X*Y'); %d*k

temp_A=sqrt(sum(W.^2,2));
temp_Bsum=repmat(sum(temp_A),[size(X,1),1]);
o=temp_Bsum./(temp_A+0.1*max(temp_A));
% o=temp_Bsum./(temp_A);
O=diag(o);

temp_obj1=norm(X'*W-Y','Fro').^2;
temp_obj2=trace(W'*O*W);
obj(iter)=temp_obj1+lambda2*temp_obj2;
end
% plot(obj)
temp_C=sqrt(sum(W.^2,2));
sita=sqrt(temp_C./sum(temp_C));