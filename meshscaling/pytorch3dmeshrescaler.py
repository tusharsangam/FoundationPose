import os
import torch
import numpy as np
from tqdm import tqdm
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras, 
    PointLights,  
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
)
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
from pytorch3d.ops import sample_points_from_meshes
# add path for demo utils functions 
import sys
import os, imageio
from skimage import img_as_ubyte
sys.path.append(os.path.abspath(''))
from utils import  read_data, MeshTransformer
import os.path as osp, numpy as np, cv2


# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Assuming camera intrinsics matrix 'K' and pose matrix 'pose' are given
# Example camera intrinsics and pose (adjust according to your data)
root_folder_path = "../demo_data/bottlevideocloser"
intrinsic_path = osp.join(root_folder_path, "cam_K.txt")

K_3x3 = np.loadtxt(intrinsic_path)  # fx, fy, cx, cy should be provided
fx, fy = K_3x3[0,0], K_3x3[1, 1]
px, py = K_3x3[0, 2], K_3x3[1, 2]

eye_in_gl, target_rgb, target_silhoutte, tgt_point_cloud = read_data(osp.join(root_folder_path, "depth", f"{str('0').zfill(5)}.png"),
                                    osp.join(root_folder_path, "masks", f"{str('0').zfill(5)}.png"),
                                    osp.join(root_folder_path, "rgb", f"{str('0').zfill(5)}.png"),
                                    fx, fy, px, py
                                     )

target_silhoutte = torch.from_numpy((target_silhoutte.astype(np.float32))).to(device)

print(f"Eye in gl {eye_in_gl}")



print("Camera Intrinsics", fx, fy, px, py)


# Set paths
obj_filename = os.path.join(root_folder_path, "mesh/mesh.obj")

# Load obj file

#mesh = load_objs_as_meshes(["data/cow_mesh/cow.obj"], device=device)
mesh:Meshes= load_objs_as_meshes([obj_filename], device=device)


# We scale normalize and center the target mesh to fit in a sphere of radius 1 
# centered at (0,0,0). (scale, center) will be used to bring the predicted mesh 
# to its original center and scale.  Note that normalizing the target mesh, 
# speeds up the optimization but is not necessary!
verts = mesh.verts_packed()
N = verts.shape[0]
center = verts.mean(0)
scale = max((verts - center).abs().max(0)[0])
mesh.offset_verts_(-center)
print(f"Scale {(1.0/float(scale))}")
# mesh.scale_verts_((1.0 / float(scale)))

# the number of different viewpoints from which we want to render the mesh.
num_views = 1

# Get a batch of viewing angles. 
elev = torch.linspace(0, 360, num_views)
azim = torch.linspace(-180, 180, num_views)

# Place a point light in front of the object. As mentioned above, the front of 
# the cow is facing the -z direction. 
lights = PointLights(device=device, location=[[0.0, 0.0, 0.0]])

# Initialize an OpenGL perspective camera that represents a batch of different 
# viewing angles. All the cameras helper methods support mixed type inputs and 
# broadcasting. So we can view the camera from the a distance of dist=2.7, and 
# then specify elevation and azimuth angles for each viewpoint as tensors. 
R, T = look_at_view_transform(eye=eye_in_gl)#look_at_view_transform(dist=2.7, elev=elev, azim=azim)
cameras = PerspectiveCameras(
                             device=device, R=R, T=T, 
                             focal_length=torch.tensor([[fx, fy]]).to(torch.float32), 
                             principal_point=torch.tensor([[px, py]]).to(torch.float32), 
                             image_size=torch.tensor([[480, 640]]), 
                             in_ndc=False
                             )

# We arbitrarily choose one particular view that will be used to visualize 
# results
#camera = PerspectiveCameras(device=device, R=R, T=T, focal_length=torch.tensor([[fx, fy]]), principal_point=torch.tensor([[px, py]]), image_size=torch.tensor([[480, 640]]))
 

raster_settings = RasterizationSettings(
    image_size=(480, 640), 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)
sigma = 1e-4
raster_settings_soft = RasterizationSettings(
    image_size=(480, 640), 
    blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
    faces_per_pixel=50, 
)

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=cameras,
        lights=lights
    )
)
renderer_silhouette = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings_soft
    ),
    shader=SoftSilhouetteShader()
)
meshes = mesh.extend(num_views)
meshes.requires_grad = False

# Render the cow mesh from each viewing angle
#R=to_origin[:3, :3].reshape(1, 3, 3), T=to_origin[:3, 3].reshape(1, 3)
rendered_images = renderer(meshes, cameras=cameras, lights=lights)

# meshes_transformed_random = meshes.clone().to(device)
# angles = np.deg2rad([90, 0, 0])
# initial_rotation = euler_angles_to_matrix(torch.from_numpy(angles).reshape(1, 3), "XYZ").to(device)
# random_tranform = Transform3d().rotate(initial_rotation).translate(torch.tensor([[0.05, 0.0, 0.0]]).to(device)).to(device)
# meshes_transformed_random = meshes_transformed_random.update_padded(random_tranform.transform_points(meshes_transformed_random.verts_padded()))
# rendered_silhoutte = renderer_silhouette(meshes_transformed_random, cameras=cameras, lights=lights)
# rendered_silhoutte = rendered_silhoutte[0, ..., 3]

#cv2.imwrite("silhoutte.png", (rendered_silhoutte.data.cpu().numpy()*255.).astype(np.uint8))
#target_rgb = cv2.imread("silhoutte.png")



transformer = MeshTransformer(meshes, renderer_silhouette, cameras, lights, target_rgb).to(device)

#BGR to RGB
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.005)
filename_output = "./optimization.gif"
writer = imageio.get_writer(filename_output, mode='I', duration=0.3)
tgt_point_cloud = torch.from_numpy(tgt_point_cloud).to(torch.float32).to(device).reshape(1, -1, 3)
min_loss, good_scale = torch.inf, 1.0
num_iterations = 1000
with tqdm(total=num_iterations) as pbar:
    for i in range(num_iterations):  # Example number of iterations
        optimizer.zero_grad()
        
        # Apply transformation and render
        rendered_images, meshes_transformed = transformer()
        loss = torch.sum((rendered_images[0, ..., 3] - transformer.image_ref) ** 2)
        chamfer_loss, _ = chamfer_distance(sample_points_from_meshes(meshes_transformed, 5000), tgt_point_cloud)
        loss += chamfer_loss
        
        # Compute loss and perform backpropagation
        loss.backward(retain_graph=True)
        optimizer.step()
        pbar.set_description(f"Processing iteration {i + 1}, loss {loss.item()}, scale {transformer.scale.item()}")
        #print(f"Loss: {loss.item()}")
        if loss.item() < min_loss:
            min_loss = loss.item()
            good_scale = transformer.scale.item()
        image = rendered_images[0, ..., 3].detach().squeeze().cpu().numpy()
        image = img_as_ubyte(image)
        writer.append_data(image)
        pbar.update(1)
writer.close()
cv2.imwrite("render.png", image)
print(f"Min loss {min_loss}, scale {good_scale}")