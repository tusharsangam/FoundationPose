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

mesh_path = "/home/tushar/Desktop/bottlenerfstudio/exports/mesh/edited/mesh.obj"

if __name__=='__main__':
  code_dir = os.path.dirname(os.path.realpath(__file__))
  

  est_refine_iter = 5
  track_refine_iter = 2
  debug = 3
  debug_dir = "debug"

  set_logging_format()
  set_seed(0)

  mesh = trimesh.load(mesh_path)

  
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  port = 2100
  context = zmq.Context()
  socket = context.socket(zmq.REP)
  socket.bind(f"tcp://*:{port}")
  print(f"Pose Estimation Server Listening on port {port}")
  while True:
      # rgb_image, raw_depth, mask, K
      req_data = socket.recv_pyobj()
      if len(req_data) == 6: 
        est_refine_iter, track_refine_iter = req_data[-2], req_data[-1]
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

        if debug>=1:
          center_pose = pose@np.linalg.inv(to_origin)
          vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
          vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
          cv2.imwrite('1.png', vis[...,::-1])
          #cv2.waitKey(1)


        if debug>=2:
          os.makedirs(f'{reader.video_dir}/track_vis', exist_ok=True)
          imageio.imwrite(f'{reader.video_dir}/track_vis/{reader.id_strs[i]}.png', vis)
        break

      print(f"Inference time {time.time() - start_time} secs")
      socket.send_pyobj((center_pose, pose, to_origin, vis))

  

