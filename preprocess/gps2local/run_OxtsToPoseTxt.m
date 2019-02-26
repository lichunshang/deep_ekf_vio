function run_OxtsToPoseTxt(base_dir)
% KITTI RAW DATA DEVELOPMENT KIT
% 
% Plots OXTS poses of a sequence
%
% Input arguments:
% base_dir .... absolute path to sequence base directory (ends with _sync)

disp('======= KITTI DevKit Demo =======');

% load oxts data
oxts = loadOxtsliteData(base_dir);

% transform to poses
pose = convertOxtsToPose(oxts);

fileID = fopen('/home/cs4li/Downloads/poses.txt','w');

T_velo_imu = [9.999976e-01 7.553071e-04 -2.035826e-03 -8.086759e-01;
             -7.854027e-04 9.998898e-01 -1.482298e-02 3.195559e-01;
              2.024406e-03 1.482454e-02 9.998881e-01 -7.997231e-01;
              0 0 0 1];
T_cam_velo = [7.967514e-03 -9.999679e-01 -8.462264e-04 -1.377769e-02;
             -2.771053e-03 8.241710e-04 -9.999958e-01 -5.542117e-02;
              9.999644e-01 7.969825e-03 -2.764397e-03 -2.918589e-01;
              0 0 0 1];
          
T_cam_imu = T_cam_velo * T_velo_imu;
    

for i=1:length(pose)
    T_C0_Ck = T_cam_imu * pose{i} * inv(T_cam_imu);
    p = T_C0_Ck';
    p = p(1:12);
    fprintf(fileID, '%.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n', ...
        p(1), p(2), p(3), p(4), p(5), p(6), p(7), p(8), p(9), p(10), p(11), p(12));
end