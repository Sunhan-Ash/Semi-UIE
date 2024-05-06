% 设置result文件夹路径
% folder = "/home/xusunhan/RAUNE/RAUNE-Net-main/results/waternet/EUVP/waternet_EUVP/single/predicted";
folder = "../result3.4/UIEBD";
% /media/xusunhan/storage/semi_log/12.20/UIEBD_1220
% folder = '/media/xusunhan/storage/semi_log/12.24/UIEBD2';
% targetfolder="/home/xusunhan/second/MySecond//data/UIEBD2/val/GT";
% 获取所有图片文件4.607
files = dir(fullfile(folder,'*.png'));

% target = dir(fullfile(targetfolder,'*.png'));

% 预分配存储结果矩阵
% results = cell(length(files), 6);
results = cell(length(files), 3);

% 循环每个图片计算UIQM
for i = 1:length(files)
    I = imread(fullfile(folder,files(i).name));
    % T = imread(fullfile(targetfolder,target(i).name));
    results{i, 1} = files(i).name;
    results{i, 2} = UIQM(I);
    results{i, 3} = UCIQE(I);
    % results{i, 4} = niqe(I);
    % results{i, 5} = psnr(I,T);
    % results{i, 6} = ssim(I,T);
end

% 转换cell数组为table，并保存为CSV
% resultsTable = cell2table(results, 'VariableNames', {'Filename', 'UIQM', 'UCIQE', 'NIQE','PSNR','SSIM'});
resultsTable = cell2table(results, 'VariableNames', {'Filename', 'UIQM', 'UCIQE'});
csvFileName = fullfile(folder, 'combined_results.csv');
writetable(resultsTable, csvFileName);

% 输出平均值结果
fprintf('Average UIQM for images in folder %s is %.3f\n', folder, mean(cell2mat(results(:,2))));
fprintf('Average UCIQE for images in folder %s is %.3f\n', folder, mean(cell2mat(results(:,3))));
% fprintf('Average NIQE for images in folder %s is %.3f\n', folder, mean(cell2mat(results(:,4))));
% fprintf('Average PSNR for images in folder %s is %.3f\n', folder, mean(cell2mat(results(:,5))));
% fprintf('Average SSIM for images in folder %s is %.3f\n', folder, mean(cell2mat(results(:,6))));


