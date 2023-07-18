import os
import sys
import numpy as np
import cv2
import open3d as o3d
from path import Path
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data_plane")
sys.path.append(DATA_PATH)


if __name__ == '__main__':
    camera_path = os.path.join(DATA_PATH, "depthData/camera")
    cameraDepth_path = os.path.join(DATA_PATH, "depthData/cameraDepth")
    cameraLabels_path = os.path.join(DATA_PATH, "depthData/cameraLabels")

    camera_filenames = os.listdir(camera_path)
    cameraDepth_filenames = os.listdir(cameraDepth_path)
    cameraLabels_filenames = os.listdir(cameraLabels_path)
    assert len(camera_filenames) == len(cameraDepth_filenames) == len(cameraLabels_filenames)

    camera_files = sorted([os.path.join(camera_path, f) for f in camera_filenames])
    cameraDepth_files = sorted([os.path.join(cameraDepth_path, f) for f in cameraDepth_filenames])
    cameraLabels_files = sorted([os.path.join(cameraLabels_path, f) for f in cameraLabels_filenames])
    assert len(camera_files) == len(cameraDepth_files) == len(cameraLabels_files)

    for data_id in range(10, len(camera_files)):
        camera0 = np.load(camera_files[data_id])
        cameraDepth0 = np.load(cameraDepth_files[data_id], allow_pickle=True)
        cameraLabels0 = np.load(cameraLabels_files[data_id], allow_pickle=True).item()['data']
        # print("\nData shape in camera folder: ", camera0.shape)
        # print("\nData shape in cameraDepth folder: ", cameraDepth0.shape)
        # print("Data in cameraDepth folder: ", cameraDepth0)
        # print("\nData shape in cameraLabels folder: ", cameraLabels0.shape)
        # print("Data in cameraLabels folder: ", cameraLabels0)
        # print("Labels of planes: ", cameraLabels0[cameraLabels0==2].shape)

        # create point cloud
        color = o3d.geometry.Image(camera0)
        depth = o3d.geometry.Image(cameraDepth0)
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        pinhole_camera_intrinsic.set_intrinsics(1024, 1024, 24, 24, 0, 0)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity = False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)

        # add gaussian noise
        sigma = (0.0035 - 0.001) * np.random.random_sample() + 0.001
        pcd_shape = np.asarray(pcd.points).shape
        gaussian_noise = np.random.normal(0, sigma, size=pcd_shape)
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) + gaussian_noise)
        assert np.asarray(pcd.points).shape[0] == cameraLabels0[cameraLabels0==2].shape[0]
        # print(sigma)
        # print(gaussian_noise.shape)

        # flip the orientation, so it looks upright, not upside-down
        pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        pcd_path = os.path.join(DATA_PATH, f"pointcloud/{data_id}.pcd")
        o3d.io.write_point_cloud(pcd_path, pcd)

        # save the images
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True) #works for me with False, on some systems needs to be true
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(os.path.join(DATA_PATH, f"images/{data_id}.png"))
        vis.destroy_window()

        # visualize the point cloud
        # o3d.visualization.draw_geometries([pcd])    


