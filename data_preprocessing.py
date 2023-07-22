import os
import sys
import numpy as np
import cv2
import open3d as o3d
from path import Path
from tqdm import tqdm
import argparse
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data_synthetic")
sys.path.append(DATA_PATH)


if __name__ == '__main__':
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Data Preprocessing')
    parser.add_argument('--data', type=str, default='plane_pcd', help='The kind of raw data')
    args = parser.parse_args()

    if args.data == "plane_pcd":
        cloud_path = os.path.join(DATA_PATH, "rawDataPlane/cameraPC")
        cloud_filename = sorted(os.listdir(cloud_path))
        cloud_files = [os.path.join(cloud_path, filename) for filename in cloud_filename]

        # loading ground truth label
        label_path = os.path.join(DATA_PATH, "rawDataPlane/cameraLabels")
        label_filename = sorted(os.listdir(label_path))
        label_files = [os.path.join(label_path, filename) for filename in label_filename]
        assert len(cloud_files) == len(label_files)

        no_noise_savepath = os.path.join(DATA_PATH, f"pcd_plane/pcd_no_noise")
        shutil.rmtree(no_noise_savepath)
        os.makedirs(no_noise_savepath)

        noise_savepath = os.path.join(DATA_PATH, f"pcd_plane/pcd_noise")
        shutil.rmtree(noise_savepath)
        os.makedirs(noise_savepath)

        label_savepath = os.path.join(DATA_PATH, f"pcd_plane/label")
        shutil.rmtree(label_savepath)
        os.makedirs(label_savepath)

        for i in range(len(cloud_files)):
            xyz = np.load(cloud_files[i], allow_pickle=True).item()['data']
            labels = np.load(label_files[i], allow_pickle=True).item()['data']
            points_num = xyz.shape[0]

            if points_num > 15000:
                # create a point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz)
                
                # find available labels
                availd_labels = labels[labels!=0]
                assert availd_labels.shape[0] == points_num
                availd_labels[availd_labels != 2] = 0
                availd_labels[availd_labels == 2] = 1

                # save the label
                label_path = os.path.join(label_savepath, f"{i}.npy")
                np.save(label_path, availd_labels)

                # define the colour
                plane_colors = np.array([[0.1, 0.1, 0.3]])
                non_plane_colors = np.array([[0.8, 0.2, 0.3]])
                np_colors = np.repeat(non_plane_colors, points_num, axis=0)
                np_colors[availd_labels==1] = plane_colors
                pcd.colors = o3d.Vector3dVector(np_colors)
                # o3d.visualization.draw_geometries([pcd])
                # break

                # save the point cloud
                pcd_path = os.path.join(no_noise_savepath, f"{i}.pcd")
                o3d.io.write_point_cloud(pcd_path, pcd)

                # add gaussian noise
                a = 0.03
                b = 0.005
                sigma = (a - b) * np.random.random_sample() + b
                pcd_shape = np.asarray(pcd.points).shape
                gaussian_noise = np.random.normal(0, sigma, size=pcd_shape)
                pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) + gaussian_noise)

                # save the noisy point cloud
                pcd_path = os.path.join(noise_savepath, f"{i}.pcd")
                o3d.io.write_point_cloud(pcd_path, pcd)

                # # visualize the point cloud
                # o3d.visualization.draw_geometries([pcd])
                # break

    elif args.data == "nonplane_pcd":
        cloud_path = os.path.join(DATA_PATH, "rawDataNonPlane/cameraPC")
        cloud_filename = sorted(os.listdir(cloud_path))
        cloud_files = [os.path.join(cloud_path, filename) for filename in cloud_filename]

        # loading ground truth label
        label_path = os.path.join(DATA_PATH, "rawDataNonPlane/cameraLabels")
        label_filename = sorted(os.listdir(label_path))
        label_files = [os.path.join(label_path, filename) for filename in label_filename]
        assert len(cloud_files) == len(label_files)

        no_noise_savepath = os.path.join(DATA_PATH, f"pcd_nonplane/pcd_no_noise")
        shutil.rmtree(no_noise_savepath)
        os.makedirs(no_noise_savepath)

        noise_savepath = os.path.join(DATA_PATH, f"pcd_nonplane/pcd_noise")
        shutil.rmtree(noise_savepath)
        os.makedirs(noise_savepath)

        label_savepath = os.path.join(DATA_PATH, f"pcd_nonplane/label")
        shutil.rmtree(label_savepath)
        os.makedirs(label_savepath)

        for i in range(len(cloud_files)):
            xyz = np.load(cloud_files[i], allow_pickle=True).item()['data']
            labels = np.load(label_files[i], allow_pickle=True).item()['data']
            points_num = xyz.shape[0]

            if points_num > 15000:
                # create a point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz)
                
                # find available labels
                availd_labels = labels[labels!=0]
                assert availd_labels.shape[0] == points_num
                availd_labels[availd_labels != 6] = 0
                availd_labels[availd_labels == 6] = 1

                # save the label
                label_path = os.path.join(label_savepath, f"{i}.npy")
                np.save(label_path, availd_labels)

                # define the colour
                plane_colors = np.array([[0.1, 0.1, 0.3]])
                non_plane_colors = np.array([[0.8, 0.2, 0.3]])
                np_colors = np.repeat(non_plane_colors, points_num, axis=0)
                np_colors[availd_labels==1] = plane_colors
                pcd.colors = o3d.Vector3dVector(np_colors)
                # o3d.visualization.draw_geometries([pcd])
                # break

                # save the point cloud
                pcd_path = os.path.join(no_noise_savepath, f"{i}.pcd")
                o3d.io.write_point_cloud(pcd_path, pcd)

                # add gaussian noise
                a = 0.03
                b = 0.005
                sigma = (a - b) * np.random.random_sample() + b
                pcd_shape = np.asarray(pcd.points).shape
                gaussian_noise = np.random.normal(0, sigma, size=pcd_shape)
                pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) + gaussian_noise)

                # save the noisy point cloud
                pcd_path = os.path.join(noise_savepath, f"{i}.pcd")
                o3d.io.write_point_cloud(pcd_path, pcd)

                # # visualize the point cloud
                # o3d.visualization.draw_geometries([pcd])
                # break

    else:
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

        for data_id in range(158, 159):
            camera0 = np.load(camera_files[data_id])
            cameraDepth0 = np.load(cameraDepth_files[data_id], allow_pickle=True)
            cameraLabels0 = np.load(cameraLabels_files[data_id], allow_pickle=True).item()['data']
            # Here we need to check the classes. if you print out 
            print(cameraLabels0) # we will get the full data. we need to see what values correspond to plane and others
            # this is because the dataset generator has many more classes (in the image on the right there is purple yellow, blue, dark blue - these are all diff)
            # basically "plane" may be class 2 again here but you will need to check it.
            # when you find out which one is a plane you can change all the others like this using numpy.
            plane_id = 2 # if this is the id of  a plane, it may be a string
            cameraLabels0[cameraLabels0 != plane_id] = 0 # we denote a non plane as a 0
            cameraLabels0[cameraLabels0 == plane_id] = 1 # enforce whatever value here to 1, a plane

            # create point cloud
            height, width = 1024, 1024
            focal_length =  24 # 0.024, 24
            vert_aperture = 15.2908
            horiz_aperture = 20.955
            
            # focal_x = height * focal_length / vert_aperture 
            # focal_y = width * focal_length / horiz_aperture 
            # focal_x = focal_length
            # focal_y = focal_length
            # focal_x = focal_length * vert_aperture / horiz_aperture
            # focal_y = focal_length * horiz_aperture / vert_aperture
            focal_x = focal_length * horiz_aperture / vert_aperture
            focal_y = focal_length * vert_aperture / horiz_aperture

            center_x = height * 0.5 
            center_y = width * 0.5
            # center_x = horiz_aperture * 0.5 
            # center_y = vert_aperture * 0.5

            color = o3d.geometry.Image(camera0)
            depth = o3d.geometry.Image(cameraDepth0)
            camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
            # camera_intrinsic.set_intrinsics(width, height, focal_x, focal_y, center_x, center_y)
            camera_intrinsic.intrinsic_matrix =  [[focal_x, 0.00, center_x] , [0.00, focal_y, center_y], [0.00, 0.00, 1.00]]
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity = False)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsic)

            if np.asarray(pcd.points).shape[0] > 40000:
                # flip the orientation, so it looks upright, not upside-down
                pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

                # # save the point cloud
                # pcd_path = os.path.join(DATA_PATH, f"pcd_no_noise/{data_id}.pcd")
                # o3d.io.write_point_cloud(pcd_path, pcd)

                # # save the images
                # vis = o3d.visualization.Visualizer()
                # vis.create_window(visible=True) #works for me with False, on some systems needs to be true
                # vis.add_geometry(pcd)
                # vis.poll_events()
                # vis.update_renderer()
                # vis.capture_screen_image(os.path.join(DATA_PATH, f"images_no_noise/{data_id}.png"))
                # vis.destroy_window()

                # # add gaussian noise
                # a = 0.03
                # b = 0.025
                # sigma = (a - b) * np.random.random_sample() + b
                # pcd_shape = np.asarray(pcd.points).shape
                # gaussian_noise = np.random.normal(0, sigma, size=pcd_shape)
                # pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) + gaussian_noise)
                # assert np.asarray(pcd.points).shape[0] == cameraLabels0[cameraLabels0==2].shape[0]

                # # save the noisy point cloud
                # pcd_path = os.path.join(DATA_PATH, f"pcd_noise/{data_id}.pcd")
                # o3d.io.write_point_cloud(pcd_path, pcd)

                # # save the images
                # vis = o3d.visualization.Visualizer()
                # vis.create_window(visible=True) #works for me with False, on some systems needs to be true
                # vis.add_geometry(pcd)
                # vis.poll_events()
                # vis.update_renderer()
                # vis.capture_screen_image(os.path.join(DATA_PATH, f"images_noise/{data_id}.png"))
                # vis.destroy_window()

                # visualize the point cloud
                o3d.visualization.draw_geometries([pcd])    


