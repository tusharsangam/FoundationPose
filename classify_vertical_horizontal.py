import trimesh
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2
import os, os.path as osp
import magnum as mn

root_dir = "./demo_data/bottlevideocloser"
mesh_path = "/home/tushar/Desktop/bottlenerfstudio/exports/mesh/edited/mesh.obj"
gripper_T_intel_path = "/home/tushar/Desktop/spot-sim2real/spot_rl_experiments/spot_rl/utils/gripper_T_intel.npy"
pose_dir = f"{root_dir}/ob_in_cam"
vis_dir = f"{root_dir}/track_vis_scaled_mesh_adjusted"
mesh = trimesh.load(mesh_path)
to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
to_origin_R = R.from_matrix(to_origin[:3, :3])


camera_pose = mn.Matrix4(np.loadtxt(f"{root_dir}/body_T_gripper.txt"))@mn.Matrix4(np.load(gripper_T_intel_path)) #mn.Matrix4.from_(mn.Matrix3(R.from_euler("zyx", [-45, -15, -50], True).as_matrix()), mn.Vector3(0.0, 0.0, 0.0))
print(R.from_matrix(np.array(camera_pose)[:3, :3]).as_euler("zyx", True))
def detect_orientation_by_z_axis_transform(pose:R, mesh_rotation:R, pose_magnum:mn.Matrix4):
    z_axis:np.ndarray = np.array([0, 0, 1])
    y_axis:np.ndarray = np.array([0, 1, 0])
    x_axis:np.ndarray = np.array([1, 0, 0])
    
    z_axis = mn.Matrix4(to_origin).inverted().transform_vector(mn.Vector3(*z_axis)).normalized()
    y_axis = mn.Matrix4(to_origin).inverted().transform_vector(mn.Vector3(*y_axis)).normalized()
    x_axis = mn.Matrix4(to_origin).inverted().transform_vector(mn.Vector3(*x_axis)).normalized()

    #breakpoint()
    
    z_axis_transformed = (camera_pose@pose_magnum).transform_vector(z_axis).normalized()
    y_axis_transformed = (camera_pose@pose_magnum).transform_vector(y_axis).normalized()
    x_axis_transformed = (camera_pose@pose_magnum).transform_vector(x_axis).normalized()
    
    
    theta = np.rad2deg(np.arccos(np.clip(np.dot( np.array(z_axis), np.array(z_axis_transformed) ), -1, 1)))
    
    gamma = np.rad2deg(np.arccos(np.clip(np.dot( np.array(y_axis), np.array(y_axis_transformed) ), -1, 1)))
    alpha = np.rad2deg(np.arccos( np.clip(np.dot( np.array(x_axis), np.array(x_axis_transformed) ), -1, 1) ))
    
    return f"vertical {format(theta, '.2f')}, {format(gamma, '.2f')}, {format(alpha, '.2f')}" if theta < 50 else f"horizontal {format(theta, '.2f')}, {format(gamma, '.2f')}, {format(alpha, '.2f')}"

for rgb_file_name in sorted(os.listdir(vis_dir), reverse=True):
    if "vis_score_" in rgb_file_name : continue
    pose_file_name = rgb_file_name.replace("png", "txt")
    pose_file_name = osp.join(pose_dir, pose_file_name)
    rgb_file_name = osp.join(vis_dir, rgb_file_name)
    image = cv2.imread(rgb_file_name)
    pose_magnum = mn.Matrix4(np.loadtxt(pose_file_name))@mn.Matrix4(to_origin).inverted()
    pose = R.from_matrix(np.loadtxt(pose_file_name)[:3, :3])
    pose = pose*to_origin_R.inv()
    text = detect_orientation_by_z_axis_transform(pose, to_origin_R, pose_magnum) + f", {osp.basename(rgb_file_name)}"
    image = cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,  1, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imshow("Orientation", image)
    keyCode = cv2.waitKey(0)
    if (keyCode & 0xFF) == ord("q"):
        break

cv2.destroyAllWindows()