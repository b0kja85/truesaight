import torch
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from ml_model.model_architecture import DeepfakeModel

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

def extract_frames(video_path: str, num_frames=80):
    """
    Extracts 80 evenly spaced frames from the video.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=np.int32)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames

def get_face_embeddings(frames):
    """
    Uses FaceNet to extract 512-d face embeddings from given frames.
    """
    embeddings = []

    for frame in frames:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face_tensor = mtcnn(img)

        if face_tensor is not None:
            with torch.no_grad():
                emb = resnet(face_tensor.unsqueeze(0))
                embeddings.append(emb.squeeze(0))

    if not embeddings:
        return None

    return torch.stack(embeddings)  # Shape: (N, 512)

def predict_from_embeddings(embeddings):
    """
    Runs your custom model on the averaged embeddings.
    """
    with torch.no_grad():
        averaged = embeddings.mean(dim=0, keepdim=True)  # Shape: (1, 512)
        output = custom_model(averaged)
        prediction = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][prediction].item()

    return prediction, round(confidence * 100, 2)

def run_deepfake_model(video_path: str):
    """
    Full pipeline: extract frames → get embeddings → predict
    """
    frames = extract_frames(video_path)
    embeddings = get_face_embeddings(frames)

    if embeddings is None:
        return 'unknown', 0.0, 0  # or raise an error

    pred, confidence = predict_from_embeddings(embeddings)

    label_map = {0: 'real', 1: 'fake'}  # Change based on your model classes
    result_label = label_map.get(pred, 'unknown')

    return result_label, confidence, len(embeddings)
