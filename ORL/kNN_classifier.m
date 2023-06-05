function  [Y_compute, Y_prob] = kNN_classifier(k,X_train, Y_train, X_test, Y_test)
% each row is a sample

if (isempty(X_train)),
   fprintf('Error: The training set is empty!\n');
   return;
end;

% Y_train and Y_test should be column vectors
if size(Y_train,2) ~= 1
    Y_train = Y_train';
end
if size(Y_test,2) ~= 1
    Y_test = Y_test';
end;

class_set = unique(Y_train);

num_test = size(X_test, 1);   
num_train = size(X_train, 1);
num_feature = size(X_test, 2);
Y_compute = zeros(num_test, 1);
Y_prob = zeros(num_test, 1);

X_train_sqr = sqrt(sum(X_train .* X_train, 2));
X_test_sqr = sqrt(sum(X_test .* X_test, 2));

for i = 1:num_test    
    sumDistance = zeros(num_train, 1);
    %for j = 1:num_train
    %    sumDistance(j) = vecdist(X_train(j, :), X_test(i, :), disttype);
    %end;
    sumDistance = vecdist(X_train, X_test(i, :), 0, X_train_sqr, X_test_sqr(i));  
    [sortDis, Index] = sort(sumDistance);
    n = hist(Y_train(Index(1:k)), class_set);
    [junk, index] = max(n);
    Y_compute(i) = class_set(index);
    Y_prob(i) = sortDis(1); 
end;

% Y_prob = Y_compute;

    %compute the error rate
    result = Y_test - Y_compute;
    index = find(result == 0);
    Y_prob = Y_compute;
    Y_compute = length(index)/num_test;
    
    
function dist = vecdist(X_train_vec, X_test_vec, disttype, X_train_sqr_vec, X_test_sqr)

switch(disttype)
case 0
    X_diff = (X_train_vec - repmat(X_test_vec, size(X_train_vec, 1), 1));
    dist = sum(X_diff .* X_diff, 2);
case 1
    plusdist = (X_train_vec + repmat(X_test_vec, size(X_train_vec, 1), 1));
    plusdist = plusdist + (plusdist == 0) * 1e-8;
    minusdist = (X_train_vec - repmat(X_test_vec, size(X_train_vec, 1), 1)); 
    dist = sum(minusdist .* minusdist ./ plusdist, 2); % chi2 distance
case 2
    dist = sum((X_train_vec .* repmat(X_test_vec, size(X_train_vec, 1), 1)), 2);    
    dist = (dist ./ X_train_sqr_vec) / X_test_sqr;
    dist = -dist; % cosine similarity, make it a distance 
end;