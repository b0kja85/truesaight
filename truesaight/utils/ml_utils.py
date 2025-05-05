import torch
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import gc
import os

from django.conf import settings
from ..models import VideoProcessing

from torchvision.transforms import Compose, Resize, ToTensor
from ml_model.model_architecture import DeepfakeModel # Import Deepfake Architecture

# Load face detection and embedding models
mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# ==========================

########################################
# Global Variable for Multiprocessing
########################################
face_detector = None

def initializer():
    """
    Initialize the global MTCNN face detector.
    """
    global face_detector, device
    # Use GPU if available; otherwise, use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    face_detector = MTCNN(keep_all=False, device=device)


# Frame sampling configuration
target_fps = 8       # Frames per second to sample
max_duration = 10    # Process up to 10 seconds per video
target_frames = 80   # Total frames per video
    
# Configuration for face detection and cropping
FACE_MARGIN = 0.2  # 20% margin around detected face (adjust as needed)

########################################
# Utility Functions
########################################

def force_resize_no_padding(image, target_size=(224, 224)):
    """
    Forcefully resize the image to target_size, ignoring aspect ratio.
    No black borders are added.
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def central_crop(image):
    """
    Return a centered square crop of the image.
    If image is not square, the largest possible centered square is returned.
    """
    h, w = image.shape[:2]
    if h > w:
        top = (h - w) // 2
        return image[top:top+w, :]
    else:
        left = (w - h) // 2
        return image[:, left:left+h]

def crop_face_with_margin(image, margin=FACE_MARGIN):
    """
    Detect the face in the image using MTCNN and return a crop of the image that
    covers the detected face expanded by a margin. If no face is detected, return
    a central crop.
    
    Args:
        image (np.array): The input image in BGR format.
        margin (float): Fractional margin to expand the detected face box.
        
    Returns:
        np.array: Cropped image.
    """
    # Convert to RGB for MTCNN
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes, _ = face_detector.detect(rgb_image)
    
    if boxes is not None and len(boxes) > 0:
        # Use the first detected face
        box = boxes[0]  # [x1, y1, x2, y2]
        w_box = box[2] - box[0]
        h_box = box[3] - box[1]
        new_x1 = max(0, int(box[0] - margin * w_box))
        new_y1 = max(0, int(box[1] - margin * h_box))
        new_x2 = min(image.shape[1], int(box[2] + margin * w_box))
        new_y2 = min(image.shape[0], int(box[3] + margin * h_box))
        crop = image[new_y1:new_y2, new_x1:new_x2]
        return crop
    else:
        # Fallback: return a centered square crop
        return central_crop(image)


def extract_face(video_path: str, video_instance: VideoProcessing):
    print(f"[DEBUG] Starting face extraction for video: {video_path}")

    mtcnn = initializer()
    if mtcnn:
        print("MTCNN initialized successfully!")
    else:
        print("Failed to initialize MTCNN.")


    try:
        cap = cv2.VideoCapture(video_path)
        print("[DEBUG] VideoCapture initialized")
    except Exception as e:
        print(f"[ERROR] Cannot open video file: {video_path}, Error: {e}")
        video_instance.status = 'failed'
        video_instance.error_message = str(e)
        video_instance.save()
        return

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file: {video_path}")
        video_instance.status = 'failed'
        video_instance.error_message = "Unable to open video"
        video_instance.save()
        return
    
    try:
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[DEBUG] Original FPS: {original_fps}")
        if original_fps <= 0:
            print("[ERROR] Invalid FPS detected")
            cap.release()
            video_instance.status = 'failed'
            video_instance.error_message = "Invalid FPS"
            video_instance.save()
            return

        total_possible_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = int(min(total_possible_frames, target_fps * max_duration))
        print(f"[DEBUG] Total frames in video: {total_possible_frames}, Frames to process: {total_frames}")

        if total_frames <= 0:
            print("[ERROR] No valid frames to process")
            video_instance.status = 'failed'
            video_instance.error_message = "No frames in video"
            video_instance.save()
            cap.release()
            return

        frame_indices = np.linspace(0, total_frames - 1, num=target_frames, dtype=int)
        print(f"[DEBUG] Frame indices: {frame_indices}")

        processed_frames = []

        media_path = os.path.join(settings.MEDIA_ROOT, 'extracted_frames', str(video_instance.pk))
        os.makedirs(media_path, exist_ok=True)
        print(f"[DEBUG] Saving frames to: {media_path}")

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            print(f"[DEBUG] Reading frame {idx}: ret={ret}, frame is None: {frame is None}")

            if not ret or frame is None:
                print(f"[WARNING] Could not read frame {idx}")
                if processed_frames:
                    processed_frames.append(processed_frames[-1].copy())
                else:
                    processed_frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
                continue

            try:
                cropped = crop_face_with_margin(frame, margin=FACE_MARGIN)
                resized = force_resize_no_padding(cropped, (224, 224))
                final_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                processed_frames.append(final_rgb)
            except Exception as fe:
                print(f"[WARNING] Face processing failed on frame {idx}: {fe}")
                processed_frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

        cap.release()

        while len(processed_frames) < target_frames:
            processed_frames.append(processed_frames[-1].copy())

        print(f"[DEBUG] Processed frames: {len(processed_frames)}")

        for idx, frame_img in enumerate(processed_frames):
            frame_path = os.path.join(media_path, f"frame_{idx:05d}.jpg")
            frame_bgr = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)
            success = cv2.imwrite(frame_path, frame_bgr)
            print(f"[DEBUG] Saving frame {idx} to {frame_path}: {'Success' if success else 'Failed'}")

        video_instance.status = 'processing'
        video_instance.processed_frames_count = len(processed_frames)
        video_instance.save()

        print(f"[DEBUG] Finished processing {video_path}")

    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        video_instance.status = 'failed'
        video_instance.error_message = str(e)
        video_instance.save()
        cap.release()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

transform = Compose([
    Resize((224, 224)),
    ToTensor()
])

# Load processed frames (assuming they're saved as RGB .jpg files)
def load_video_frames_as_tensor(folder_path, seq_len=40):
    frames = []
    for i in range(seq_len):
        img_path = os.path.join(folder_path, f"frame_{i:05d}.jpg")
        image = Image.open(img_path).convert("RGB")
        frames.append(transform(image))

    video_tensor = torch.stack(frames)  # Shape: (seq_len, 3, 224, 224)
    video_tensor = video_tensor.unsqueeze(0)  # Add batch dim: (1, seq_len, 3, 224, 224)
    return video_tensor

def create_chain_graph(seq_len: int) -> torch.Tensor:
    edge_list = []
    for i in range(seq_len - 1):
        edge_list.extend([[i, i+1], [i+1, i]])
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()

def create_batched_edge_index(base_edge_index, batch_size, num_nodes, device):
    edge_index = base_edge_index.clone().repeat(1, batch_size)
    offsets = torch.arange(batch_size, device=device) * num_nodes
    num_edges_per_sample = base_edge_index.size(1)
    offsets = offsets.unsqueeze(0).repeat(2, num_edges_per_sample)
    edge_index += offsets
    return edge_index

def run_deepfake_model(video_path: str, instance_id: int):
    seq_len = 40
    dropout_rate = 0.5
    custom_model = DeepfakeModel(seq_len=seq_len, dropout_rate=dropout_rate).to(device)

    # Load the model from the saved state
    model_path = 'ml_model/trained-model.pth'
    custom_model.load_state_dict(torch.load(model_path, map_location=device))
    custom_model.eval()  # Set model to evaluation mode
    
    threshold = 0.5
    
    frames_dir = os.path.join(settings.MEDIA_ROOT, 'extracted_frames', str(instance_id))
    input_tensor = load_video_frames_as_tensor(frames_dir, seq_len=seq_len).to(device)
    batch_size = input_tensor.size(0)

    with torch.no_grad():
        base_edge_index = create_chain_graph(seq_len).to(device)
        batched_edge_index = create_batched_edge_index(base_edge_index, batch_size, seq_len, device)
        output = custom_model(input_tensor, batched_edge_index)  
        score = torch.sigmoid(output).item()
        
        # Use the dynamic threshold here
        label = 'real' if score >= threshold else 'fake'
        print(f"[DEBUG] Prediction Score: {score:.4f}, Label: {label}, Threshold: {threshold:.4f}")
        return label, score