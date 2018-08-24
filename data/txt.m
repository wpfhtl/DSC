clear;
clc;

root = '/home/wpf/data/SRD/Train/shadow/';
%root = '/home/wpf/data/ISTD/train_argu/train_A/';

res = dir(fullfile(root,'*.jpg'));

fid = fopen('./SRD/train.txt','w');


num_image = length(res);

% random order
 choose = randperm(num_image)';
 res = res(choose,:);

for i=1:num_image
    
   
    fprintf(fid,'%s', res(i).name);
    fprintf(fid,'\n');
    
end

fclose(fid);
