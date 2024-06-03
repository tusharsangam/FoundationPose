# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from spotsim2realdatareader import *
import zmq
from tqdm import tqdm
from yacs.config import CfgNode as CN

#mesh_path = "/home/tushar/Desktop/penguine_nerf/exports/mesh/edited/mesh.obj"
#mesh_path = "/home/tushar/Desktop/bottle_nerf/exports/mesh/edited/mesh.obj"
mesh_dir = "../../spot_rl_experiments/weights/mesh"
config_path = "../../spot_rl_experiments/configs/config.yaml"
cn = CN()
cn.set_new_allowed(True)
cn.merge_from_file(config_path)
POSE_PORT = cn.POSE_PORT

mesh_folder_paths = os.listdir(mesh_dir)
mesh_folder_paths = [os.path.join(mesh_dir, object_name) for object_name in mesh_folder_paths]

meshes_memory_cache = {}

for mesh_folder_path in tqdm(mesh_folder_paths, desc="Loading meshes in memory for fast access..", total=len(mesh_folder_paths)):
  mesh_path = os.path.join(mesh_folder_path, "mesh.obj")
  object_name = mesh_folder_path.split("/")[-1]
  
  mesh = trimesh.load(mesh_path)
  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
  mesh_ori = mesh.copy()
  mesh = mesh.copy()
  
  model_normals = mesh.vertex_normals.copy()
  max_xyz = mesh.vertices.max(axis=0)
  min_xyz = mesh.vertices.min(axis=0)
  model_center = (min_xyz+max_xyz)/2
  mesh.vertices = mesh.vertices - model_center.reshape(1,3)

  model_pts = mesh.vertices
  diameter = compute_mesh_diameter(model_pts=mesh.vertices, n_sample=10000)
  vox_size = max(diameter/20.0, 0.003)
  
  pcd = toOpen3dCloud(model_pts, normals=model_normals)
  pcd = pcd.voxel_down_sample(vox_size)
  max_xyz = np.asarray(pcd.points).max(axis=0)
  min_xyz = np.asarray(pcd.points).min(axis=0)

  pts = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device='cuda')
  normals = F.normalize(torch.tensor(np.asarray(pcd.normals), dtype=torch.float32, device='cuda'), dim=-1)
  mesh_path = f'/tmp/{uuid.uuid4()}.obj'
  mesh.export(mesh_path)
  mesh_tensors = make_mesh_tensors(mesh)

  meshes_memory_cache[object_name] = (mesh_ori, to_origin, extents, bbox, mesh, model_center, diameter, max_xyz, min_xyz, pts, normals, mesh_path, mesh_tensors) 

if __name__=='__main__':
  code_dir = os.path.dirname(os.path.realpath(__file__))
  

  est_refine_iter = 5
  track_refine_iter = 2
  debug = 0
  debug_dir = "debug"

  set_logging_format()
  set_seed(0)
  

  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  mesh_path = os.path.join(mesh_folder_paths[0], "mesh.obj")
  mesh = trimesh.load(mesh_path)
  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
  
  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  #port = 2100
  context = zmq.Context()
  socket = context.socket(zmq.REP)
  socket.bind(f"tcp://*:{POSE_PORT}")
  print(f"Pose Estimation Server Listening on port {POSE_PORT}")
  while True:
      # rgb_image, raw_depth, mask, K
      req_data = socket.recv_pyobj()
      if len(req_data) == 8: 
        est_refine_iter, track_refine_iter = req_data[-2], req_data[-1]
      if len(req_data) > 4:
        #TODO: Select mesh & its scaling factor based on these parameters
        image_src, object_name = req_data[4], req_data[5]
        for mesh_folder_path in mesh_folder_paths:
          if mesh_folder_path.split("/")[-1] in object_name:
            mesh_vals = meshes_memory_cache[mesh_folder_path.split("/")[-1]]
            break
        # mesh_path = os.path.join(mesh_folder_path, "mesh.obj")
        # mesh = trimesh.load(mesh_path)
        # to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        # bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
        mesh_ori, to_origin, extents, bbox, mesh, model_center, diameter, max_xyz, min_xyz, pts, normals, mesh_path, mesh_tensors = mesh_vals
        est.reset_object_fast(mesh_ori=mesh_ori, mesh=mesh, model_center=model_center, diameter=diameter, max_xyz=max_xyz, min_xyz=min_xyz, pts=pts, normals=normals, mesh_path=mesh_path, mesh_tensors=mesh_tensors)

      reader = YcbineoatReaderModified(req_data, shorter_side=None, zfar=np.inf)
      
      start_time = time.time()

      for i in range(len(reader.color_files)):
        logging.info(f'i:{i}')
        color = reader.get_color(i)
        depth = reader.get_depth(i)
        if i==0:
          mask = reader.get_mask(0).astype(bool)
          pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=est_refine_iter)

          if debug>=3:
            m = mesh.copy()
            m.apply_transform(pose)
            m.export(f'{debug_dir}/model_tf.obj')
            xyz_map = depth2xyzmap(depth, reader.K)
            valid = depth>=0.1
            pcd = toOpen3dCloud(xyz_map[valid], color[valid])
            o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
        else:
          pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=track_refine_iter)

        os.makedirs(f'{reader.video_dir}/ob_in_cam', exist_ok=True)
        np.savetxt(f'{reader.video_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))

        if True:
          center_pose = pose@np.linalg.inv(to_origin)
          vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
          vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
          #cv2.imwrite('1.png', vis[...,::-1])
          #cv2.waitKey(1)


        if debug>=2:
          os.makedirs(f'{reader.video_dir}/track_vis', exist_ok=True)
          imageio.imwrite(f'{reader.video_dir}/track_vis/{reader.id_strs[i]}.png', vis)
        break

      print(f"Inference time {time.time() - start_time} secs")
      socket.send_pyobj((center_pose, pose, to_origin, vis))

  

