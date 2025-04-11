from .ml_utils import run_deepfake_model

def process_video_task(video_id):
    from .models import VideoProcessing
    video_obj = VideoProcessing.objects.get(pk=video_id)
    video_obj.status = 'processing'
    video_obj.save()

    try:
        result, confidence, frame_count = run_deepfake_model(video_obj.video_file.path)

        video_obj.result = result
        video_obj.confidence_score = confidence
        video_obj.processed_frames_count = frame_count
        video_obj.status = 'completed'
        video_obj.save()
        
    except Exception as e:
        video_obj.status = 'failed'
        video_obj.error_message = str(e)
        video_obj.save()
        print(f"[Error] Processing failed for Video ID {video_id}: {e}")
