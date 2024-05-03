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
    BlendParams
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
from utils import  read_data, MeshTransformer, fill_holes
import os.path as osp, numpy as np, cv2
from pytorch_msssim import MS_SSIM
from torch.optim.swa_utils import AveragedModel, SWALR
import atexit

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
#fill_holes(obj_filename)
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
# mesh.scale_verts_((1.0 / float(scale)))

# the number of different viewpoints from which we want to render the mesh.
num_views = 1
 
lights = PointLights(device=device, location=[[0.0, 0.0, 0.0]])



def compute_elevation_azimuth(x, y, z):
    """
    Compute the elevation and azimuth of a vector given its x, y, z components.
    
    Args:
    x (float): X component of the vector.
    y (float): Y component of the vector.
    z (float): Z component of the vector.
    
    Returns:
    tuple: (elevation, azimuth) in degrees.
    """
    # Convert to PyTorch tensors
    x, y, z = map(torch.tensor, (x, y, z))
    
    # Calculate the radius
    r = torch.sqrt(x**2 + y**2 + z**2)
    
    # Elevation
    elevation = torch.asin(z / r)
    
    # Azimuth
    azimuth = torch.atan2(y, x)
    
    # Convert from radians to degrees
    elevation_deg = torch.rad2deg(elevation)
    azimuth_deg = torch.rad2deg(azimuth)
    
    return elevation_deg.item(), azimuth_deg.item()
elev, azim = compute_elevation_azimuth(*eye_in_gl[0])
print(elev, azim)
#R, T = look_at_view_transform(eye=eye_in_gl)
R, T = look_at_view_transform(dist=eye_in_gl[0][-1], elev=elev, azim=azim, degrees=True)
cameras = PerspectiveCameras(
                             device=device, R=R, T=T, 
                             focal_length=torch.tensor([[fx, fy]]).to(torch.float32), 
                             principal_point=torch.tensor([[px, py]]).to(torch.float32), 
                             image_size=torch.tensor([[480, 640]]), 
                             in_ndc=False
                             )

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
        lights=lights,
        #blend_params=BlendParams(background_color=(0, 0, 0))
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


def close_writers():
    global writer_sil
    global writer_rgb
    writer_sil.close()
    writer_rgb.close()

atexit.register(close_writers)


transformer = MeshTransformer(meshes, renderer_silhouette, renderer, cameras, lights, target_rgb, eye_in_gl).to(device)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.005) #best 0.005
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1, verbose=False)

ms_ssim_rgb = MS_SSIM(data_range=1., channel=3, size_average=True)
ms_ssim_silhoutte = MS_SSIM(data_range=1., channel=1, size_average=True)

writer_sil = imageio.get_writer("./optimization_sil.gif", mode='I', duration=0.3)
writer_rgb = imageio.get_writer("./optimization_rgb.gif", mode='I', duration=0.3)

tgt_point_cloud = torch.from_numpy(tgt_point_cloud).to(torch.float32).to(device).reshape(1, -1, 3)
min_loss, good_scale = torch.inf, 1.0
num_iterations = 800

text_orig = (50, 50)
target_rgb_show = (target_rgb.copy()*255.).astype(np.uint8)
target_rgb_show_ = cv2.putText(target_rgb_show.copy(), "Real Target view", text_orig, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
#cv2.namedWindow("Training")

with tqdm(total=num_iterations) as pbar:
    for i in range(num_iterations):  # Example number of iterations
        optimizer.zero_grad()
        
        # Apply transformation and render
        rendered_images_sil, rendered_images_rgb, meshes_transformed = transformer()
    
        loss = 0.0
        
        loss += torch.sum((rendered_images_sil[0, ..., 3] - transformer.image_ref) ** 2)
        loss += torch.sum((rendered_images_rgb[0, ..., :3] - transformer.image_ref_rgb) ** 2)
        loss += chamfer_distance(sample_points_from_meshes(meshes_transformed, 5000), tgt_point_cloud)[0]
        loss += 1. - ms_ssim_silhoutte(rendered_images_sil[..., 3].unsqueeze(-1).permute(0, 3, 1, 2), transformer.image_ref.unsqueeze(0).unsqueeze(-1).permute(0, 3, 1, 2)) 
        loss += 1. - ms_ssim_rgb(rendered_images_rgb[..., :3].permute(0, 3, 1, 2), transformer.image_ref_rgb.unsqueeze(0).permute(0, 3, 1, 2)) 
        
        
        # Compute loss and perform backpropagation
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        # 
        if loss.item() < min_loss:
            min_loss = loss.item()
            good_scale = transformer.scale.item()
        
        
        image_sil = img_as_ubyte(rendered_images_sil[0, ..., 3].detach().squeeze().cpu().numpy())
        image = img_as_ubyte(rendered_images_rgb[0, ..., :3].detach().squeeze().cpu().numpy())
        
        image_ = cv2.putText(image.copy(), f"Rendering at t={i}, scale {transformer.scale.item():.2f}", text_orig, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
        overlaid_image = np.clip(image[..., ::-1]+target_rgb_show, 0, 255).astype(np.uint8)
        overlaid_image = cv2.putText(overlaid_image, "Real+Rendering", text_orig, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
        image_show = np.hstack([image_[..., ::-1], target_rgb_show_, overlaid_image])
        writer_sil.append_data(image_sil)
        writer_rgb.append_data(image_show[..., ::-1])
        #cv2.imshow("Training", image_show)
        #cv2.waitKey(1)
        
        pbar.set_description(f"Processing iteration {i + 1}, loss {loss.item()}, scale {transformer.scale.item()}")
        pbar.update(1)
        
        cv2.imwrite("render.png", image_show)
        #break
#cv2.destroyAllWindows()
#swa_model.update_parameters(transformer)
print(f"Min loss {min_loss}, scale {good_scale}")

writer_sil.close()
writer_rgb.close()

