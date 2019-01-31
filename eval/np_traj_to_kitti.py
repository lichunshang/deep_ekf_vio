import setup

data_dir = os.path.join(config.save_path, "trajectory_results")

sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]


def write_trj(file_handle, pose):
    m = transformations.quaternion_matrix(pose[3:7])
    m[0, 3] = pose[0]
    m[1, 3] = pose[1]
    m[2, 3] = pose[2]

    m = np.concatenate(m[0:3])
    m = m.astype(np.float32)
    file_handle.write(" ".join(["%g" % i for i in list(m)]) + "\n")


for seq in sequences:
    np_file = os.path.join(data_dir, "trajectory_%s.npy" % seq)
    kitti_file = open(os.path.join(data_dir, "kitti_evals", "%s.txt" % seq), "w")
    trajectory = np.load(np_file)

    for i in range(0, trajectory.shape[0]):
        write_trj(kitti_file, trajectory[i])

    kitti_file.close()
