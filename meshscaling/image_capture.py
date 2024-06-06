from spot_wrapper.spot import Spot, image_response_to_cv2
import numpy as np, cv2, os, os.path as osp
import magnum as mn, time, shutil
import rospy
from spot_rl.envs.skill_manager import SpotSkillManager
from copy import deepcopy

cam_device = "gripper"
cam_index = 1 if cam_device == "intel" else 0
save_name_path = f"penguin_scan_anchor_{cam_device}"

path_to_save = f"../demo_data/{save_name_path}"
path_to_gripper_T_intel = "/home/tushar/Desktop/spot-sim2real/spot_rl_experiments/spot_rl/utils/gripper_T_intel.npy"
def record_transform(spot:Spot):
    rospy.set_param("is_gripper_blocked", cam_index)
    image_resps = spot.get_hand_image() #IntelImages
    image_responses = [image_response_to_cv2(image_rep) for image_rep in image_resps]    
    intrinsics = image_resps[0].source.pinhole.intrinsics
    fx, fy = intrinsics.focal_length.x, intrinsics.focal_length.y
    px, py = intrinsics.principal_point.x, intrinsics.principal_point.y
    rospy.set_param("is_gripper_blocked", 0)
    image_resps_gripper = spot.get_hand_image()
    K = np.identity(3).astype(np.float32)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, -1] = px
    K[1, -1] = py
    K_path = osp.join(path_to_save, "cam_K.txt")
    np.savetxt(K_path, K)

    tree = image_resps_gripper[0].shot.transforms_snapshot
    hand_T_gripper:mn.Matrix4 = spot.get_magnum_Matrix4_spot_a_T_b("arm0.link_wr1", "hand_color_image_sensor", tree)
    gripper_T_intel:mn.Matrix4 = mn.Matrix4(np.load(path_to_gripper_T_intel)) if cam_index == 1 else mn.Matrix4(np.eye(4))

    body_T_hand:mn.Matrix4 = spot.get_magnum_Matrix4_spot_a_T_b("body", "link_wr1")
    body_T_intel = body_T_hand@(hand_T_gripper@gripper_T_intel)
    transform:np.ndarray = np.array(body_T_intel)
    transform_save_path = osp.join(path_to_save, "body_T_intel.txt")
    np.savetxt(transform_save_path, transform)
    print(transform)
    rospy.set_param("is_gripper_blocked", cam_index)

if __name__ == "__main__":
    # try:
    #     shutil.rmtree(path_to_save)
    # except:
    #     pass
    os.makedirs(path_to_save, exist_ok=True)
    spot_skill_manager = SpotSkillManager()
    spot_skill_manager.spot.stand()
    spot_skill_manager.spot.open_gripper()
    gaz_arm_angles = deepcopy(spot_skill_manager.pick_config.GAZE_ARM_JOINT_ANGLES)
    gaz_arm_angles[-2] = 75 if cam_index == 1 else gaz_arm_angles[-2]
    #spot_skill_manager.spot.set_arm_joint_positions(np.deg2rad(gaz_arm_angles), 1)
    
    spot = spot_skill_manager.spot
    i = 0
    rgb_path = f"{path_to_save}/rgb"
    depth_path = f"{path_to_save}/depth"
    os.makedirs(rgb_path, exist_ok=True)
    os.makedirs(depth_path, exist_ok=True)
    record_transform(spot)
    cv2.namedWindow('Intel Image', cv2.WINDOW_NORMAL)
    while True:
        
        image_resps = spot.get_hand_image()
        image_responses = [image_response_to_cv2(image_rep) for image_rep in image_resps]
        cv2.imshow("Intel Image", image_responses[0])
        key = cv2.waitKey(1) & 0xFF
    
        # Check if q is pressed
        if key == ord('q'):
            break  # Break the loop if q is pressed
        elif key == 13:
            print(f"Saving {i} saved")
            rgb_path_ = osp.join(rgb_path, f"{str(i).zfill(5)}.png")
            cv2.imwrite(rgb_path_, image_responses[0])
            
            depth_path_ = osp.join(depth_path, f"{str(i).zfill(5)}.png")
            cv2.imwrite(depth_path_, image_responses[1])
            i+=1

    cv2.destroyAllWindows()
