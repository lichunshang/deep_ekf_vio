close all;
% clear;
clc;

% test to see how the raw dataset from GPS matches with odometry dataset
t1_odom_ld = importdata('/home/cs4li/Dev/KITTI/dataset/poses/06.txt');
t2_gen_ld = importdata('/home/cs4li/Dev/KITTI/dataset/2011_09_30/2011_09_30_drive_0020_extract/oxts/poses.txt');

t1_0 = eye(4, 4);
t1_odom = zeros(4, 4, size(t1_odom_ld, 1));
for i=1:size(t1_odom_ld, 1)
    t1_odom(:, :, i) = inv(t1_0) * [reshape(t1_odom_ld(i, :), 4, 3)'; 0 0 0 1];
end

t2_gen = zeros(4, 4, size(t2_gen_ld, 1));
% t2_0 = eye(4, 4);
t2_0 = [reshape(t2_gen_ld(1, :), 4, 3)'; 0 0 0 1];
for i=1:size(t2_gen_ld, 1)
    t2_gen(:, :, i) = inv(t2_0) * [reshape(t2_gen_ld(i, :), 4, 3)'; 0 0 0 1];
end

figure(1);
hold on;
plot(squeeze(t1_odom(1, 4, :)), squeeze(t1_odom(2, 4, :)));
plot(squeeze(t2_gen(1, 4, :)), squeeze(t2_gen(2, 4, :)));
legend('odom', 'gps');
xlabel('x'); ylabel('y');
title('XY');
axis equal;

figure(2);
hold on;
plot(squeeze(t1_odom(1, 4, :)), squeeze(t1_odom(3, 4, :)));
plot(squeeze(t2_gen(1, 4, :)), squeeze(t2_gen(3, 4, :)));
legend('odom', 'gps');
xlabel('X'); ylabel('z');
title('XZ');
axis equal;

figure(3);
hold on;
plot(squeeze(t1_odom(2, 4, :)), squeeze(t1_odom(3, 4, :)));
plot(squeeze(t2_gen(2, 4, :)), squeeze(t2_gen(3, 4, :)));
legend('odom', 'gps');
xlabel('y'); ylabel('z');
title('YZ');
axis equal;