function run_OxtsToPoseTxt(base_dir)
% KITTI RAW DATA DEVELOPMENT KIT
% 
% Plots OXTS poses of a sequence
%
% Input arguments:
% base_dir .... absolute path to sequence base directory (ends with _sync)

disp('Converting GPS data from: ');
disp(base_dir)

% load oxts data
oxts = loadOxtsliteData(base_dir);

% transform to poses
poses = convertOxtsToPose(oxts);

file = fopen(strcat(base_dir, '/oxts/poses.txt'),'w');
% T_velo_imu = importdata(strcat(base_dir, '/../T_velo_imu.txt'));
% T_cam_velo = importdata(strcat(base_dir, '/../T_cam_velo.txt'));

% T_velo_imu = [9.999976e-01 7.553071e-04 -2.035826e-03 -8.086759e-01;
%              -7.854027e-04 9.998898e-01 -1.482298e-02 3.195559e-01;
%               2.024406e-03 1.482454e-02 9.998881e-01 -7.997231e-01;
%               0 0 0 1];
% T_cam_velo = [7.027555e-03 -9.999753e-01 2.599616e-05 -7.137748e-03;
%              -2.254837e-03 -4.184312e-05 -9.999975e-01 -7.482656e-02;
%               9.999728e-01 7.027479e-03 -2.255075e-03 -3.336324e-01;
%               0 0 0 1];
          
% T_cam_imu = T_cam_velo * T_velo_imu;
posmodes = zeros(size(poses, 3), 1);
velmodes = zeros(size(poses, 3), 1);
rotmodes = zeros(size(poses, 3), 1);

for i=1:size(poses, 3)
%     poses(:,:,i) = T_cam_imu * poses(:,:,i) * inv(T_cam_imu);
    p = poses(:,:,i)';
    p = p(1:12);
    fprintf(file, '%.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n', ...
        p(1), p(2), p(3), p(4), p(5), p(6), p(7), p(8), p(9), p(10), p(11), p(12));
    
    posmodes(i) = oxts{i}(end - 2);
    velmodes(i) = oxts{i}(end - 1);
    rotmodes(i) = oxts{i}(end);
end

fprintf('posmodes: min=%.5f max=%.5f ave=%.5f\n', min(posmodes), max(posmodes), mean(posmodes));
fprintf('velmodes: min=%.5f max=%.5f ave=%.5f\n', min(velmodes), max(velmodes), mean(velmodes));
fprintf('rotmodes: min=%.5f max=%.5f ave=%.5f\n', min(rotmodes), max(rotmodes), mean(rotmodes));

% Save the figures
figure('visible','off'); clf;
hold on;
plot(squeeze(poses(1, 4, :)), squeeze(poses(2, 4, :)));
xlabel('x'); ylabel('y');
title('XY');
axis equal;
grid minor;
saveas(gcf, strcat(base_dir, '/oxts/XY.png'))

figure('visible','off'); clf;
hold on;
plot(squeeze(poses(1, 4, :)), squeeze(poses(3, 4, :)));
xlabel('x'); ylabel('z');
title('XZ');
axis equal;
grid minor;
saveas(gcf, strcat(base_dir, '/oxts/XZ.png'))

figure('visible','off'); clf;
hold on;
plot(squeeze(poses(2, 4, :)), squeeze(poses(3, 4, :)));
xlabel('y'); ylabel('z');
title('YZ');
axis equal;
grid minor;
saveas(gcf, strcat(base_dir, '/oxts/YZ.png'))

figure('visible','off'); clf;
hold on;
plot(posmodes);
xlabel('frames []'); ylabel('mode');
title('Posmode');
grid minor;
saveas(gcf, strcat(base_dir, '/oxts/posmodes.png'))

figure('visible','off'); clf;
hold on;
plot(velmodes);
xlabel('frames []'); ylabel('mode');
title('Velmode');
grid minor;
saveas(gcf, strcat(base_dir, '/oxts/velmodes.png'))

figure('visible','off'); clf;
hold on;
plot(rotmodes);
xlabel('frames []'); ylabel('mode');
title('Rotmode');
grid minor;
saveas(gcf, strcat(base_dir, '/oxts/rotmodes.png'))