clear;
clc;

%root = '/home/wpf/data/SBU-shadow/SBUTrain4KRecoveredSmall/';
root = '/home/wpf/data/SBU-shadow/SBU-Test/';

res = dir(fullfile(strcat(root,'ShadowImages'),'*.jpg'));
gt = dir(fullfile(strcat(root,'ShadowMasks'),'*.png'));

fid = fopen('/home/wpf/pkg/DSC/data/SBU/test.txt','w');

for i=1:length(res)
    
   
    fprintf(fid,'%s/%s', 'ShadowImages', res(i).name);
    %fprintf(fid,' %s/%s', 'ShadowMasks', gt(i).name);
    fprintf(fid,'\n');
    
end

fclose(fid);
