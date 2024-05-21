import math
import os

import cv2
import numpy as np
import open3d as o3d
import rospy
import torch
import zmq
from scipy.spatial.transform import Rotation
from segment_anything import SamPredictor, build_sam
from transformers import Owlv2ForObjectDetection, Owlv2Processor

socket = None
model, processor = None, None
sam = None
device = "cuda"

class FocalLength:
        def __init__(self, x, y):
            self.x = x
            self.y = y
class PrincipalPoint:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

class Intrinsics:
    def __init__(self, fx:float, fy:float, cx:float, cy:float):
        self.focal_length:FocalLength = FocalLength(fx, fy)
        self.principal_point:PrincipalPoint = PrincipalPoint(cx, cy) 

def connect_socket(port):
    global socket
    if socket is None:
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(f"tcp://localhost:{port}")
        print(f"Socket Connected at {port}")
    else:
        print("Connected socket found")
    return socket

def load_model(model_name="owlvit", device="cpu"):
    global model
    global processor
    global sam
    if model_name == "owlvit":
        print("Loading OwlVit2")
        model = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16-ensemble"
        ).to(device)
        processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        # run segment anything (SAM)
    if model_name == "sam":
        print("Loading SAM")
        sam = SamPredictor(
            build_sam(
                checkpoint="/home/tushar/Desktop/spot-sim2real/spot_rl_experiments/weights/sam_vit_h_4b8939.pth"
            ).to(device)
        )


# load_model("owlvit", device)
# load_model("sam", device)
connect_socket(2100)

def detect(img, text_queries, score_threshold, device):
    global model
    global processor

    if model is None or processor is None:
        load_model("owlvit", device)

    text_queries = text_queries
    text_queries = text_queries.split(",")
    size = max(img.shape[:2])
    target_sizes = torch.Tensor([[size, size]])
    device = model.device
    inputs = processor(text=text_queries, images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    outputs.logits = outputs.logits.cpu()
    outputs.pred_boxes = outputs.pred_boxes.cpu()
    results = processor.post_process_object_detection(
        outputs=outputs, target_sizes=target_sizes
    )
    boxes, scores, labels = (
        results[0]["boxes"],
        results[0]["scores"],
        results[0]["labels"],
    )

    result_labels = []
    for box, score, label in zip(boxes, scores, labels):
        box = [int(i) for i in box.tolist()]
        if score < score_threshold:
            continue
        result_labels.append((box, text_queries[label.item()], score))
    result_labels.sort(key=lambda x: x[-1], reverse=True)
    return result_labels


def segment(image, boxes, size, device):

    global sam
    if sam is None:
        load_model("sam", device)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam.set_image(image)

    # H, W = size[1], size[0]

    for i in range(boxes.shape[0]):
        boxes[i] = torch.Tensor(boxes[i])

    boxes = torch.tensor(boxes, device=sam.device)

    transformed_boxes = sam.transform.apply_boxes_torch(boxes, image.shape[:2])

    masks, _, _ = sam.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    return masks

def get_mask(rgb_image:np.ndarray, bbox:list, object_name:str, device:str="cuda")->np.ndarray:
    h, w, _ = rgb_image.shape
    if bbox is None:
        label = detect(rgb_image, object_name, 0.01, device)
        print("prediction label", label)
        bbox = label[0][0]  # Example bounding box
    x1, y1, x2, y2 = bbox
    masks = segment(rgb_image, np.array([[x1, y1, x2, y2]]), [h, w], device)
    mask = masks[0, 0].cpu().numpy()
    h, w = mask.shape
    mask_image = np.dstack([mask, mask, mask]).reshape(h, w, 3).astype(np.uint8)*255
    return mask_image

def pose_estimation(
    rgb_image:np.ndarray, depth_raw:np.ndarray, bbox:list, object_name:str, cam_intrinsics:Intrinsics, mask:np.ndarray=None, device:str="cuda"
) -> None:
    # load_models(device)
    global socket
    assert socket is not None, "socket is not connected to server"
    fx = cam_intrinsics.focal_length.x
    fy = cam_intrinsics.focal_length.y
    cx = cam_intrinsics.principal_point.x
    cy = cam_intrinsics.principal_point.y
    K = np.identity(3).astype(np.float32)
    K[0,0] = fx
    K[1,1] = fy
    K[0, -1] = cx
    K[1, -1] = cy

    if tuple(rgb_image.shape) != (480, 640, 3):
        rgb_image = cv2.resize(rgb_image, (640, 480))
        if depth_raw is not None:
            depth_raw = cv2.resize(depth_raw, (640, 480))
    
    if mask is None:
        mask_image = get_mask(rgb_image, bbox, object_name, device)
    else:
        mask_image = mask

    cv2.imwrite("masked_image.png", mask_image)
    est_refine_iter = 5
    track_refine_iter = 2
    socket.send_pyobj((rgb_image, depth_raw, mask_image, K.astype(np.double), est_refine_iter, track_refine_iter))
    pose = socket.recv_pyobj()
    print(f"Response from server {pose}")



if __name__ == "__main__":
    import os.path as osp
    from tqdm import tqdm
    from perception_and_utils.utils.generic_utils import map_user_input_to_boolean
    #load static data
    #data_root_path = "/fsx-siro/sangamtushar/FoundationPose/demo_data/bottleposevideo"
    data_root_path = "./demo_data/bottle_scan_anchor_intel"
    intrinsics = [
        383.2665100097656,
        383.2665100097656,
        324.305419921875,
        236.64828491210938,
    ]

    # rgb_path = osp.join(data_root_path, "rgb", "00000.png") 
    # depth_path = osp.join(data_root_path, "depth", "00000.png")
    # mask_path = osp.join(data_root_path, "masks", "00000.png")
    # rgb = cv2.imread(rgb_path)
    # depth = cv2.imread(depth_path, -1)
    # mask = cv2.imread(mask_path)
    # camera_intrinsics:Intrinsics = Intrinsics(*intrinsics)
    # pose_estimation(rgb, depth, None, "bottle", camera_intrinsics, mask)
    
    
    rgb_filenames = os.listdir(osp.join(data_root_path, "rgb"))
    rgb_filenames.sort()
    #print(rgb_filenames)
    for rgb_file_name in tqdm(rgb_filenames, desc="Generating mask..", total=len(rgb_filenames)):
        print(osp.join(data_root_path, "rgb", rgb_file_name))
        rgb_image = cv2.imread(osp.join(data_root_path, "rgb", rgb_file_name))
        mask_image_file_name = osp.join(data_root_path, "masks", rgb_file_name)
        try:
            mask_image = get_mask(rgb_image, None, "bottle", "cuda")
            cv2.imshow("Mask", mask_image)
            cv2.waitKey(1)
            save_or_not = map_user_input_to_boolean("Save this image ?")
            if save_or_not:
                cv2.imwrite(mask_image_file_name, mask_image)
            else:
                raise Exception("Don't save the image object not clear")
        except Exception as e:
            print(f"Error occured {e}")
            os.unlink(osp.join(data_root_path, "rgb", rgb_file_name))
            os.unlink(osp.join(data_root_path, "depth", rgb_file_name))
    cv2.destroyAllWindows()
        
    
    #pose_estimation(rgb, depth, None, "bottle", camera_intrinsics, "cuda")

    #Otherwise do it using spot
    #from spot_wrapper.spot import Spot, SpotCamIds, image_response_to_cv2
    