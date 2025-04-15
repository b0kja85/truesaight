from .ml_utils import run_deepfake_model
from .ml_utils import extract_face  
from ..models import VideoProcessing

def process_video_task(video_id):
    try:
        # Get video object by primary key (video_id)
        video_obj = VideoProcessing.objects.get(pk=video_id)
        video_obj.status = 'processing'
        video_obj.save()

        # Get the file path
        video_path = video_obj.video_file.path  # Make sure video_file is a FileField

        # STEP 1: Extract frames and save to disk
        face_frames_path = extract_face(video_path, video_obj)

        # STEP 2: Run the deepfake detection model
        result, confidence = run_deepfake_model(video_path=face_frames_path, instance_id=video_id)

        # STEP 3: Save results to the VideoProcessing model
        video_obj.result = result
        video_obj.confidence_score = confidence
        video_obj.status = 'completed'
        video_obj.save()

    except Exception as e:
        # In case of an error, set status to failed and save error message
        video_obj.status = 'failed'
        video_obj.error_message = str(e)  # Ensure `error_message` exists in your model
        video_obj.save()
        print(f"[Error] Processing failed for Video ID {video_id}: {e}")
