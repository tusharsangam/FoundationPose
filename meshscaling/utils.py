# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
import cv2
import os.path as osp
import numpy as np
import torch
from pytorch3d.transforms import Transform3d
from pytorch3d.structures import Meshes
from skimage import img_as_ubyte
import open3d as o3d

def image_grid(
    images,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
):
    """
    A util function for plotting a grid of images.

    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.

    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            # only render RGB channels
            ax.imshow(im[..., :3])
        else:
            # only render Alpha channel
            ax.imshow(im[..., 3])
        if not show_axes:
            ax.set_axis_off()

def sample_patch_around_point(
    cx: int, cy: int, depth_raw: np.ndarray, patch_size: int = 5
) -> int:
    """
    Samples a median depth in 5x5 patch around given x, y (pixel location in depth image array) as center in raw depth image
    """
    h, w = depth_raw.shape
    x1, x2 = cx - patch_size // 2, cx + patch_size // 2
    y1, y2 = cy - patch_size // 2, cy + patch_size // 2
    x1, x2 = np.clip([x1, x2], 0, w)
    y1, y2 = np.clip([y1, y2], 0, h)
    deph_patch = depth_raw[y1:y2, x1:x2]
    deph_patch = deph_patch[deph_patch > 0]
    return np.median(deph_patch)


def plot_pointcloud(points, title=""):
    # Sample points uniformly from the surface of the mesh.
    # x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
    x, y, z = points[:, 0], points[:, 1], points[:, -1]    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)
    plt.savefig(f"{title}.png")

def farthest_point_sampling(points, num_samples):
    """
    Downsamples N X 3 point cloud data to num_samples X 3 where num_samples <= N, selects farthest points
    """
    if num_samples <= points.shape[0]:
        return points
    farthest_pts = np.zeros((num_samples, 3))
    farthest_pts[0] = points[np.random.randint(len(points))]
    distances = np.linalg.norm(points - farthest_pts[0], axis=1)
    for i in range(1, num_samples):
        farthest_pts[i] = points[np.argmax(distances)]
        distances = np.minimum(
            distances, np.linalg.norm(points - farthest_pts[i], axis=1)
        )
    return farthest_pts

def generate_point_cloud(
    image, depth, mask, fx=383.883, fy=383.883, cx=324.092, cy=238.042
):
    """
    Generate a point cloud from an RGB image, depth image, and bounding box.

    Parameters:
    - rgb_image: RGB image as a numpy array.
    - depth_image: Depth image as a numpy array.
    - bbox: Bounding box as a tuple (x_min, y_min, x_max, y_max).
    - fx, fy: Focal lengths of the camera in x and y dimensions.
    - cx, cy: Optical center of the camera in x and y dimensions.

    Returns:
    - point_cloud: Open3D point cloud object.
    """
    rows, cols = np.where(mask)

    # Get the depth values at these indices
    depth_values = depth[rows, cols]

    depth_values = depth_values.astype(np.float32)
    depth_values /= 1000.0

    # Compute the 3D coordinates
    X = (cols - cx) * depth_values / fx
    Y = (rows - cy) * depth_values / fy
    Z = depth_values

    # Combine the X, Y, Z coordinates into a single N x 3 array
    points_3D = np.vstack((X, Y, Z)).T

    # Optional: Filter out points with a depth of 0 or below a certain threshold
    valid_depth_indices = depth_values > 0  # Example threshold
    print(
        f"Total vertices: {len(points_3D)}, Corrupt vertices: {len(points_3D) - len(valid_depth_indices)}"
    )
    points_3D = points_3D[valid_depth_indices]

    print(f"3D point cloud shape {points_3D.shape}")
    colors = image[rows, cols].reshape(-1, 3) / 255.0
    colors = colors[valid_depth_indices][..., ::-1]
    points_3D[:, 1] *= -1.0
    points_3D[:, -1]*= -1.0
    #plot_pointcloud(points_3D, "tgt_pointcloud")
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_3D)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    point_cloud, ind = point_cloud.remove_radius_outlier(nb_points=5, radius=0.05)
    point_cloud, ind = point_cloud.remove_statistical_outlier(nb_neighbors=5, std_ratio=2.0)
    o3d.io.write_point_cloud("tgt.ply", point_cloud)
    points_3D = np.array(point_cloud.points)
    points_3D = farthest_point_sampling(points_3D, 5000)
    return points_3D

def read_data(depth_path, object_mask_path, rgb_image_path, fx, fy, px, py):
    assert osp.exists(depth_path), f"{depth_path} doesn't exists"
    assert osp.exists(object_mask_path), f"{object_mask_path} doesn't exists"
    assert osp.exists(rgb_image_path), f"{object_mask_path} doesn't exists"
    depth_image = cv2.imread(depth_path, -1)
    object_mask = cv2.imread(object_mask_path)
    object_mask = np.where(object_mask > 0, 1, 0)
    depth_image_masked = depth_image*object_mask[..., 0].astype(depth_image.dtype)

    non_zero_indices = np.nonzero(depth_image_masked)
    # Calculate the bounding box coordinates
    y_min, y_max = non_zero_indices[0].min(), non_zero_indices[0].max()
    x_min, x_max = non_zero_indices[1].min(), non_zero_indices[1].max()

    # Format as (x1, y1, x2, y2)
    rgb_image = cv2.imread(rgb_image_path)
    rgb_image = rgb_image*(object_mask).astype(rgb_image.dtype) #object mask is 0-1
    rgb_image[object_mask[..., 0] == 0.] = [255, 255, 255]
    #rgb_image = np.where(rgb_image == 0, 255, rgb_image).astype(np.uint8)
    #rgb_image_with_bbox = cv2.rectangle(rgb_image.copy(), (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
    #cv2.imwrite("rgb_image_with_bbox.png", rgb_image)
    rgb_image = rgb_image.astype(np.float32)/255.
    
    cx, cy = (x_min + x_max) / 2., (y_min + y_max)/2.
    Z = sample_patch_around_point(int(cx), int(cy), depth_image_masked)*1e-3
    
    # print(fx, fy, cx, cy)
    # Get 3D point
    X = (cx - px) * Z / fx
    Y = (cy - py) * Z / fy
    point_at_center_in_camera_frame = np.array([X, Y, Z], dtype=np.float32)
    camera_point_in_center_frame = -1.*point_at_center_in_camera_frame
    camera_point_in_center_frame_in_gl_system = np.array([camera_point_in_center_frame[0], -1*camera_point_in_center_frame[1], -1.*camera_point_in_center_frame[-1]], dtype=np.float32)
    
    points_3d = generate_point_cloud(cv2.imread(rgb_image_path), 
                                     cv2.imread(depth_path, -1), 
                                     cv2.imread(object_mask_path)[..., 0], 
                                     fx, fy, px, py)

    return camera_point_in_center_frame_in_gl_system.reshape(1, 3).tolist(), rgb_image, object_mask[..., 0], points_3d

def fill_holes(obje_file_path):
    import trimesh
    mesh = trimesh.load(obje_file_path)
    mesh.fill_holes()
    mesh.export(obje_file_path)

from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix
class MeshTransformer(torch.nn.Module):
    def __init__(self, meshes, renderer_silhoutte, rgb_renderer, cameras, lights, image_ref, cam_eye):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.renderer_silhoutte = renderer_silhoutte
        self.renderer_rgb = rgb_renderer
        self.cameras = cameras
        self.lights = lights
        
        # Get the silhouette of the reference RGB image by finding all non-white pixel values. 
        image_ref_torch = torch.from_numpy((image_ref.copy()[..., :3].max(-1) != 1).astype(np.float32))
        self.register_buffer('image_ref', image_ref_torch)

        
        image_ref_rgb_torch = torch.from_numpy(image_ref.copy()[..., :3][..., ::-1].astype(np.float32))
        self.register_buffer('image_ref_rgb', image_ref_rgb_torch)

        cv2.imwrite("target_image_rgb.png", (image_ref*255.).astype(np.uint8))
       
        self.scale = torch.nn.Parameter(torch.tensor([1.0]).to(self.meshes.device), requires_grad=True)
        self.translate = torch.nn.Parameter(torch.tensor([[0.0, 0.0, 0.0]]).to(self.meshes.device), requires_grad=True)
        self.rotate_6d = matrix_to_rotation_6d(torch.eye(3).reshape(1, 3, 3))
        self.rotate_6d = torch.nn.Parameter(self.rotate_6d.to(self.meshes.device), requires_grad=True)
        #self.rotate = torch.nn.Parameter(torch.tensor([[0., 0., 0.]]).to(self.meshes.device), requires_grad=True)
        #self.rotate = torch.nn.Parameter(torch.eye(3).reshape(1, 3, 3).to(self.meshes.device), requires_grad=True)
    
        self.initial_correction_transformation(cam_eye)
    
    def initial_correction_transformation(self, cam_eye):
        
        #Orient the object upside down to reduce rotation parameter stress
        # angles = np.deg2rad([90, 0, 0])
        # initial_rotation = euler_angles_to_matrix(torch.from_numpy(angles).reshape(1, 3), "XYZ").to(self.meshes.device)
        # transformed_verts = Transform3d().rotate(initial_rotation).to(self.meshes.device).transform_points(self.meshes.verts_padded())
        # self.meshes = self.meshes.update_padded(transformed_verts)
        
        # angles = np.deg2rad([0, 180, 0])
        # initial_rotation = euler_angles_to_matrix(torch.from_numpy(angles).reshape(1, 3), "XYZ").to(self.meshes.device)
        # transformed_verts = Transform3d().rotate(initial_rotation).to(self.meshes.device).transform_points(self.meshes.verts_padded())
        # self.meshes = self.meshes.update_padded(transformed_verts)
        
        #transformed_verts = Transform3d().translate(-1.*torch.tensor(cam_eye).to(self.meshes.device)).to(self.meshes.device).transform_points(self.meshes.verts_padded())
        #self.meshes = self.meshes.update_padded(transformed_verts)

        angles = np.deg2rad([0, 50, 0])
        initial_rotation = euler_angles_to_matrix(torch.from_numpy(angles).reshape(1, 3), "XYZ").to(self.meshes.device)
        transformed_verts = Transform3d().rotate(initial_rotation).to(self.meshes.device).transform_points(self.meshes.verts_padded())
        self.meshes = self.meshes.update_padded(transformed_verts)

    
    def forward(self):
        
        # Render the image using the updated camera position. Based on the new position of the 
        # camera we calculate the rotation and translation matrices
        #transform3d = Transform3d(matrix=self.transformation_matrix, device=self.meshes.device
        meshes = self.meshes.clone()
        rotation_mat = rotation_6d_to_matrix(self.rotate_6d).to(self.meshes.device)
        transformed_verts = Transform3d().rotate(rotation_mat).translate(self.translate).scale(self.scale).to(self.meshes.device).transform_points(meshes.verts_padded())
        #transformed_verts = Transform3d().translate(self.translate).scale(self.scale).to(self.meshes.device).transform_points(meshes.verts_padded())
        meshes = meshes.update_padded(transformed_verts)
        image_sil = self.renderer_silhoutte(meshes_world=meshes, cameras=self.cameras, lights=self.lights)
        image_rgb = self.renderer_rgb(meshes_world=meshes, cameras=self.cameras, lights=self.lights)
        #print(self.translate, self.scale)
        
        return image_sil, image_rgb, meshes