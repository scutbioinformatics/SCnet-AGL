function [Y]=FeaNorm_ljy(Y)

%Y=k*n  n -->samples

temp_Y=Y.^2;
sum_Y=sqrt(sum(temp_Y,1));
Y=Y./repmat(sum_Y,[size(Y,1),1]);

end