import torch
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from ml_model.model_architecture import DeepfakeModel # Import Deepfake Architecture
from ml_model.losses import CombinedLoss # Import the Combined Loss
import logging # For error logging
import gc
import os

from django.conf import settings
from .models import VideoProcessing

# Load face detection and embedding models
mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Recreate model instance
custom_model = DeepfakeModel()

# Load state dict
state_dict = torch.load('ml_model/trained-model.pth', map_location=torch.device('cpu'))
custom_model.load_state_dict(state_dict)

# Set to evaluation mode
custom_model.eval()


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

def main():
    seq_len = 40
    dropout_rate = 0.5
    model = DeepfakeModel(seq_len=seq_len, dropout_rate=dropout_rate).to(device)

    transform = Compose([
        Resize((224, 224)),
        ToTensor()
    ])
    
    # Load data
    test_dataset = DeepfakeDataset(
        root_dir="/content/Deepfake-Thesis/data/Final-data/Testing",
        transform=transform,
        seq_len=seq_len
    )

    train_labels = [sample[0] for sample in train_dataset.labels]
    num_pos = sum(train_labels)
    num_neg = len(train_labels) - num_pos
    ratio = num_neg / num_pos if num_pos > 0 else 1.0
    pos_weight = torch.tensor([ratio], dtype=torch.float32).to(device)
    print(f"Computed pos_weight (from training set): {pos_weight.item():.4f}")

    criterion = CombinedLoss(
        bce_weight=0.6,
        jsd_weight=0.4,
        pos_weight=pos_weight
    )

    batch_size = 8
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Reduced to avoid overloading memory
        pin_memory=True,
        drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,  # Reduced as well
        pin_memory=True,
        drop_last=True
    )

    base_edge_index = create_chain_graph(seq_len).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    num_epochs = 20
    best_auc = 0.0

    print("Starting training...")
    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_acc, train_auc, train_f1, train_recall, train_frr, train_gar, train_precision = train_epoch(
            model, train_dataloader, criterion, optimizer, device, base_edge_index, seq_len, grad_clip=5.0
        )
        val_loss, val_acc, val_auc, val_f1, val_recall, val_frr, val_gar, val_precision = evaluate_model(
            model, test_dataloader, criterion, device, base_edge_index, seq_len
        )
        scheduler.step(val_loss)
        epoch_time = time.time() - start_time
        
        results = {
            'Epoch': epoch + 1,
            'Training': {
                'Training Loss': train_loss,
                'Training Accuracy': train_acc,
                'Training AUC': train_auc,
                'Training F1-Score': train_f1,
                'Training Recall': train_recall,
                'Training FRR': train_frr,
                'Training GAR': train_gar,
                'Training Precision': train_precision
            },
            'Testing': {
                'Val Loss': val_loss,
                'Val Accuracy': val_acc,
                'Val AUC': val_auc,
                'Val F1-Score': val_f1,
                'Val Recall': val_recall,
                'Val FRR': val_frr,
                'Val GAR': val_gar,
                'Val Precision': val_precision
            },
            'Epoch Time': epoch_time
        }

        save_model_and_result(
            model, 
            results, 
            model_filename=f"epoch-{epoch+1}-efficientgatgru-v1.pt", 
            results_filename=f"epoch-{epoch+1}-efficientgatgru-v1.json"
        )

        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train AUC: {train_auc:.4f} | "
              f"Train F1: {train_f1:.4f} | Train Recall: {train_recall:.4f} | Train FRR: {train_frr:.4f} | "
              f"Train GAR: {train_gar:.4f} | Train Precision: {train_precision:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f} | "
              f"Val Recall: {val_recall:.4f} | Val FRR: {val_frr:.4f} | Val GAR: {val_gar:.4f} | Val Precision: {val_precision:.4f}")
        print(f"Epoch Time: {epoch_time / 60:.2f} minutes")

        if val_auc > best_auc:
            best_auc = val_auc
            best_model_path = os.path.join(drive_output_dir, "models", f"best_efficientgatgru-v1_model_epoch_{epoch+1}_{current_date}.pt")
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model updated and saved to {best_model_path}")

    print("Training completed. Saving final model...")
    final_model_path = os.path.join(drive_output_dir, "models", f"deepfake_model_final_{current_date}.pt")
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")

import os
import cv2
import numpy as np
import torch
import logging
import gc
from django.conf import settings
from .models import VideoProcessing

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

def run_deepfake_model(video_path: str):
    return