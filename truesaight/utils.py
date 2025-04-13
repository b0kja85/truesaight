from .ml_utils import run_deepfake_model
from .ml_utils import extract_face  # or wherever 

def process_video_task(video_id):
    from .models import VideoProcessing
    import os

    video_obj = VideoProcessing.objects.get(pk=video_id)
    video_obj.status = 'processing'
    video_obj.save()

    try:
        video_path = video_obj.video_file.path

        # 🔍 STEP 1: Extract frames and save to disk
        extract_face(video_path, video_obj)

        # # 🔬 STEP 2: Run the model
        # # result, confidence, frame_count = run_deepfake_model(video_path)

        result = 'catch'
        confidence = 0.69

        

        # ✅ STEP 3: Save results
        video_obj.result = result
        video_obj.confidence_score = confidence
        video_obj.status = 'completed'
        video_obj.save()
        
    except Exception as e:
        video_obj.status = 'failed'
        video_obj.error_message = str(e)
        video_obj.save()
        print(f"[Error] Processing failed for Video ID {video_id}: {e}")
