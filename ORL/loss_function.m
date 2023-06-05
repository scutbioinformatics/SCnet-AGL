function [Z,U,obj]=loss_function(data,num_cluster,alpha,C,step)

% data=d*n
% alpha=1;

%[idx,C] =  kmeans(data',num_cluster,'MaxIter',1000,'Replicates',50);

zu=zeros(size(data,2),num_cluster);

for i=1:size(data,2)
    temp_zu1=repmat(data(:,i),[1,num_cluster])-C';
    temp_zu1=temp_zu1.^2;
    temp_zu2=sum(temp_zu1);
    zu(i,:)=temp_zu2;
end
temp_zu3=(1+(zu./alpha)).^((alpha+1)./(-2));

temp_zu4=repmat(sum(temp_zu3,2),[1,num_cluster]);
q=temp_zu3./(temp_zu4);

f=sum(q);
temp_q2=q.^2;
temp_q3=temp_q2./(repmat(f,[size(data,2),1]));
temp_q4=repmat(sum(temp_q3,2),[1,num_cluster]);

p=temp_q3./(temp_q4);


p_q=p-q;
delta_Z=zeros(size(data));
for i=1:size(data,2)
    temp_zi=zeros(size(data,1),1);
    for j=1:num_cluster
        zi_uj=data(:,i)-C(j,:)';
        temp_pqzu=p_q(i,j)*zi_uj;
        
        temp_ziuj=1+(sum(zi_uj.^2)./alpha);
        temp_zi=temp_zi+(temp_pqzu./temp_ziuj);
        
    end
    temp_zi=((alpha+1)./alpha)*temp_zi;
    delta_Z(:,i)=temp_zi;
end

delta_U=zeros(size(data,1),num_cluster);
for j=1:num_cluster
    temp_uj=zeros(size(data,1),1);
    for i=1:size(data,2)
        zi_uj2=data(:,i)-C(j,:)';
        temp_pqzu2=p_q(i,j)*zi_uj2;
        
        temp_ziuj2=1+(sum(zi_uj2.^2)./alpha);
        temp_uj=temp_uj+(temp_pqzu2./temp_ziuj2);

    end
    temp_uj=((alpha+1)./(-1*alpha))*temp_uj;
    delta_U(:,j)=temp_uj;
end

U=C'+ step*delta_U;
U=U';
Z=data+step*delta_Z;

obj=sum(sum(p.*log(p./q)));