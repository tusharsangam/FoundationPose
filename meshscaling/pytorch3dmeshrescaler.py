import os
import torch
import numpy as np
from tqdm import tqdm
import open3d as o3d
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.transforms import euler_angles_to_matrix, rotation_6d_to_matrix

# Data structures and functions for rendering
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.transforms import Transform3d
from pytorch3d.renderer import (
    camera_position_from_spherical_angles,
    look_at_view_transform,
    look_at_rotation,
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
    point_mesh_face_distance,
    point_mesh_edge_distance
)
from pytorch3d.ops import sample_points_from_meshes, iterative_closest_point, corresponding_points_alignment
# add path for demo utils functions 
import sys
import os, imageio
from skimage import img_as_ubyte
sys.path.append(os.path.abspath(''))
from utils import  read_data, MeshTransformer, segment, save_point_cloud, make_3d_bbox, compute_bounding_box_size
import os.path as osp, numpy as np, cv2
from pytorch_msssim import MS_SSIM
from scipy.spatial.transform import Rotation as R_
import atexit

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Assuming camera intrinsics matrix 'K' and pose matrix 'pose' are given
# Example camera intrinsics and pose (adjust according to your data)
root_folder_path = "../demo_data/penguin_scan_anchor_gripper"
intrinsic_path = osp.join(root_folder_path, "cam_K.txt")

K_3x3 = np.loadtxt(intrinsic_path).astype(np.double)  # fx, fy, cx, cy should be provided
fx, fy = K_3x3[0,0], K_3x3[1, 1]
px, py = K_3x3[0, 2], K_3x3[1, 2]


camera_position, lookat, (cx, cy, depth), target_rgb, target_silhoutte, tgt_point_cloud = read_data(
                                    osp.join(root_folder_path, "depth", f"{str('0').zfill(5)}.png"),
                                    osp.join(root_folder_path, "masks", f"{str('0').zfill(5)}.png"),
                                    osp.join(root_folder_path, "rgb", f"{str('0').zfill(5)}.png"),
                                    osp.join(root_folder_path, "body_T_intel.txt"),
                                    fx, fy, px, py
                                     )

target_silhoutte = torch.from_numpy((target_silhoutte.astype(np.float32))).to(device)

tgt_points = torch.from_numpy(tgt_point_cloud).to(torch.float32).to(device).reshape(1, -1, 3)
tgt_point_cloud = Pointclouds(tgt_points)



print("Camera Intrinsics", fx, fy, px, py)


# Set paths
obj_filename = os.path.join(root_folder_path, "mesh/mesh.obj")

# Load obj file
#fill_holes(obj_filename)
#mesh = load_objs_as_meshes(["data/cow_mesh/cow.obj"], device=device)
mesh:Meshes= load_objs_as_meshes([obj_filename], device=device)


verts = mesh.verts_packed()
N = verts.shape[0]
center = verts.mean(0)
scale = max((verts - center).abs().max(0)[0])
mesh.offset_verts_(-center)


# the number of different viewpoints from which we want to render the mesh.
num_views = 1
 
lights = PointLights(device=device, location=[[0.0, 0.0, 0.0]])



R, T = look_at_view_transform(eye=torch.from_numpy(camera_position).float().reshape(1, 3), at=torch.from_numpy(np.zeros((3,))).float().reshape(1, 3))
print(f"Camera position given {camera_position}, calculated is {-torch.bmm(R.transpose(1, 2), T.unsqueeze(2)).squeeze(2)}")
cameras:PerspectiveCameras = PerspectiveCameras(
                             device=device, R=R, T=T, 
                             focal_length=torch.tensor([[fx, fy]]).to(torch.float32), 
                             principal_point=torch.tensor([[px, py]]).to(torch.float32), 
                             image_size=torch.tensor([[480, 640]]), 
                             in_ndc=False
                             )

centroid = tgt_points[0].mean(dim=0)
centroid_in_world = cameras.get_world_to_view_transform().inverse().transform_points(centroid.reshape(1, 3))
initial_translation = centroid_in_world #cameras.unproject_points(torch.tensor([[cx, cy, depth]], dtype=torch.float32).to(device), True)
initial_translation.requires_grad_(False)

#initial_translation = cameras.get_world_to_view_transform().inverse().to(device).transform_points(initial_translation)
print(f"Initial translation after world _view_ transform {initial_translation}")
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

angles = np.deg2rad([90, 0, 0])
initial_rotation = euler_angles_to_matrix(torch.from_numpy(angles).reshape(1, 3), "XYZ").to(device).float()
#initial_translation = torch.from_numpy(initial_translation).float().reshape(1, 3).to(device)
transformer = MeshTransformer(meshes, renderer_silhouette, renderer, cameras, lights, target_rgb, initial_translation, initial_rotation).to(device)


optimizer = torch.optim.Adam([
                                {'params':transformer.scale, 'name':'scale', 'lr':0.05},
                                #{'params':transformer.translate, 'name':'translate', 'lr':0.00005},
                                #{'params':transformer.rotate_6d, 'name':'rotate_6d', 'lr':0.005},
                            ], lr=0.005) #best 0.005

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100) #(optimizer, step_size=200, gamma=0.1, verbose=False)

ms_ssim_rgb = MS_SSIM(data_range=1., channel=3, size_average=True)
ms_ssim_silhoutte = MS_SSIM(data_range=1., channel=1, size_average=True)

writer_sil = imageio.get_writer("./optimization_sil.gif", mode='I', duration=0.3)
writer_rgb = imageio.get_writer("./optimization_rgb.gif", mode='I', duration=0.3)


min_loss, good_scale, best_iter = torch.inf, 1.0, 1.0
num_iterations = 150

text_orig = (50, 50)
target_rgb_show = (target_rgb.copy()*255.).astype(np.uint8)
target_rgb_show_ = cv2.putText(target_rgb_show.copy(), "Real Target view", text_orig, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)

with torch.no_grad():
    rendered_images_sil, rendered_images_rgb, meshes_transformed = transformer.forward()
    predicted_points = sample_points_from_meshes(meshes_transformed, 5000) #in world space
    predicted_points = cameras.get_world_to_view_transform().transform_points(predicted_points)
    save_point_cloud(predicted_points, tgt_points, "before_icp.ply")
    R_init, T_init = None, None
    box_3d_predicted = make_3d_bbox(predicted_points)
    box_3d_target = make_3d_bbox(tgt_points)
    sRT = corresponding_points_alignment(box_3d_predicted, box_3d_target, estimate_scale=False)
    transformer.initial_correction_transformation(sRT.T, sRT.R, sRT.s)
    predicted_points = sample_points_from_meshes(transformer.meshes, 5000) #in world space
    predicted_points = cameras.get_world_to_view_transform().transform_points(predicted_points)
    save_point_cloud(predicted_points, tgt_points, "after_icp.ply")
    # icp_solution = iterative_closest_point(predicted_points, tgt_points, estimate_scale=True)
    # if icp_solution.converged:
    #     R_init, T_init, S_init = icp_solution.RTs.R, icp_solution.RTs.T, icp_solution.RTs.s
    #     transformer.initial_correction_transformation(None, None, S_init)
    #     print(f"s_init {S_init}")
    #     predicted_points = sample_points_from_meshes(transformer.meshes, 5000) #in world space
    #     predicted_points = cameras.get_world_to_view_transform().transform_points(predicted_points)
    #     save_point_cloud(predicted_points, tgt_points, "after_icp.ply")
        

        #transformer = MeshTransformer(meshes, renderer_silhouette, renderer, cameras, lights, target_rgb, T_init, R_init).to(device)

#cv2.namedWindow("Training")
reg_loss_weight = 10.0
min_allowable_scale = compute_bounding_box_size(tgt_points.squeeze(0)) / compute_bounding_box_size(predicted_points.squeeze(0))
min_allowable_scale += 0.02
min_allowable_scale = torch.tensor([min_allowable_scale.max().item()]).float().to(device)
print(f"Possible good scale {min_allowable_scale}")
#min_allowable_scale = torch.tensor([0.9]).float().to(device)
with tqdm(total=num_iterations) as pbar:
    for i in range(num_iterations):  # Example number of iterations
        optimizer.zero_grad()
        
        # Apply transformation and render
        rendered_images_sil, rendered_images_rgb, meshes_transformed = transformer.forward()
    
        loss = 0.0
        loss += torch.mean((rendered_images_sil[0, ..., 3] - transformer.image_ref) ** 2)
        loss += torch.mean((rendered_images_rgb[0, ..., :3] - transformer.image_ref_rgb) ** 2)
        predicted_points = sample_points_from_meshes(meshes_transformed, 5000) #in world space
        predicted_points = cameras.get_world_to_view_transform().transform_points(predicted_points)
        loss += chamfer_distance(predicted_points, tgt_points)[0]
        loss += (1. - ms_ssim_silhoutte(rendered_images_sil[..., 3].unsqueeze(-1).permute(0, 3, 1, 2), transformer.image_ref.unsqueeze(0).unsqueeze(-1).permute(0, 3, 1, 2))) 
        loss += (1. - ms_ssim_rgb(rendered_images_rgb[..., :3].permute(0, 3, 1, 2), transformer.image_ref_rgb.unsqueeze(0).permute(0, 3, 1, 2))) 
        # #loss += mesh_laplacian_smoothing(meshes_transformed)
        # #loss += mesh_edge_loss(meshes_transformed)
        # #loss += mesh_normal_consistency(meshes_transformed)
        # loss += point_mesh_edge_distance(meshes_transformed, tgt_point_cloud)
        #loss += point_mesh_face_distance(meshes_transformed, tgt_point_cloud)
        
        
        reg_loss = 0.0
        reg_loss += torch.nn.functional.relu( min_allowable_scale - (0.1 + 1.9 * torch.sigmoid(transformer.scale)) ).sum()
        #reg_loss += 10*torch.nn.functional.mse_loss(rotation_6d_to_matrix(transformer.rotate_6d), initial_rotation)
        #reg_loss += torch.nn.functional.mse_loss(transformer.translate, torch.zeros((2,), dtype=torch.float32).to(device))
        loss += reg_loss_weight*reg_loss
        
        
        # Compute loss and perform backpropagation
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            #pass
            #transformer.scale.data = torch.clamp(transformer.scale.data, 0.1, 2.0)
            scale = 0.1 + 1.9 * torch.sigmoid(torch.tensor(transformer.scale.item()))
        if loss.item() < min_loss:
            min_loss = loss.item()
            good_scale = scale
            best_iter = i + 1
        
        
        
        image_sil = img_as_ubyte(rendered_images_sil[0, ..., 3].detach().squeeze().cpu().numpy())

        image = img_as_ubyte(rendered_images_rgb[0, ..., :3].detach().squeeze().cpu().numpy())
        
        image_ = cv2.putText(image.copy(), f"Rendering at t={i}, scale {scale:.2f}", text_orig, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
        overlaid_image = np.clip(image[..., ::-1]+target_rgb_show, 0, 255).astype(np.uint8)
        overlaid_image = cv2.putText(overlaid_image, "Real+Rendering", text_orig, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
        image_show = np.hstack([image_[..., ::-1], target_rgb_show_, overlaid_image])
        writer_sil.append_data(image_sil)
        writer_rgb.append_data(image_show[..., ::-1])
        #cv2.imshow("Training", image_show)
        #cv2.waitKey(1)
        
        pbar.set_description(f"iteration {i + 1}, loss {loss.item():.2f}, scale {scale:.2f}, {reg_loss.item():.2f}")
        pbar.update(1)
        if i == 0:
            cv2.imwrite("render0.png", image_show)
        else:
            cv2.imwrite("render.png", image_show)
        #break
# without backprop
# with torch.no_grad():
#     scale_range = np.linspace(0.01, 2.0, num_iterations).astype(np.float32)
#     scale_range = torch.from_numpy(scale_range).float().to(device)
#     with tqdm(total=num_iterations) as pbar:
#         for i, scale_step in enumerate(scale_range):  # Example number of iterations
            
#             # Apply transformation and render
#             #transformer.scale[0].data = scale_step.data
#             transformer.scale[0] = scale_step
#             rendered_images_sil, rendered_images_rgb, meshes_transformed = transformer.forward(None)
        
#             loss = 0.0
            
#             loss += 0.25*torch.mean((rendered_images_sil[0, ..., 3] - transformer.image_ref) ** 2)
#             loss += 0.25*torch.mean((rendered_images_rgb[0, ..., :3] - transformer.image_ref_rgb) ** 2)
#             #loss += chamfer_distance(sample_points_from_meshes(meshes_transformed, 5000), tgt_points)[0]
#             loss += 0.25*(1. - ms_ssim_silhoutte(rendered_images_sil[..., 3].unsqueeze(-1).permute(0, 3, 1, 2), transformer.image_ref.unsqueeze(0).unsqueeze(-1).permute(0, 3, 1, 2))) 
#             loss += 0.25*(1. - ms_ssim_rgb(rendered_images_rgb[..., :3].permute(0, 3, 1, 2), transformer.image_ref_rgb.unsqueeze(0).permute(0, 3, 1, 2))) 
#             #loss += mesh_laplacian_smoothing(meshes_transformed)
#             #loss += mesh_edge_loss(meshes_transformed)
#             #loss += mesh_normal_consistency(meshes_transformed)
#             #loss += point_mesh_edge_distance(meshes_transformed, tgt_point_cloud)
#             #loss += point_mesh_face_distance(meshes_transformed, tgt_point_cloud)
            
            
#             if loss.item() < min_loss:
#                 min_loss = loss.item()
#                 good_scale = transformer.scale.item()
            
            
#             image_sil = img_as_ubyte(rendered_images_sil[0, ..., 3].detach().squeeze().cpu().numpy())
#             image = img_as_ubyte(rendered_images_rgb[0, ..., :3].detach().squeeze().cpu().numpy())
            
#             image_ = cv2.putText(image.copy(), f"Rendering at t={i}, scale {transformer.scale.item():.2f}", text_orig, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
#             overlaid_image = np.clip(image[..., ::-1]+target_rgb_show, 0, 255).astype(np.uint8)
#             overlaid_image = cv2.putText(overlaid_image, "Real+Rendering", text_orig, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
#             image_show = np.hstack([image_[..., ::-1], target_rgb_show_, overlaid_image])
#             writer_sil.append_data(image_sil)
#             writer_rgb.append_data(image_show[..., ::-1])
#             #cv2.imshow("Training", image_show)
#             #cv2.waitKey(1)
            
#             pbar.set_description(f"Processing iteration {i + 1}, loss {loss.item()}, scale {transformer.scale.item()}")
#             pbar.update(1)
            
#             cv2.imwrite("render.png", image_show)
#             #break
# cv2.destroyAllWindows()
# #swa_model.update_parameters(transformer)
print(f"Min loss {min_loss}, scale {good_scale}, best_iter {best_iter}")

writer_sil.close()
writer_rgb.close()

