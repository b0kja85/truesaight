import time
from datetime import timedelta
from .ml_utils import run_deepfake_model
from .models import VideoProcessing

def process_video_task(video_id):
    """
    Given the ID of a VideoProcessing instance,
    extract frames, run the ML model, and update the instance.
    """
    video_task = VideoProcessing.objects.get(id=video_id)

    video_task.status = 'processing'
    video_task.save(update_fields=['status'])

    start = time.time()
    result, confidence, frames_count = run_deepfake_model(video_task.video_file.path)

    duration = time.time() - start

    video_task.result = result
    video_task.confidence_score = confidence
    video_task.processed_frames_count = frames_count
    video_task.processing_duration = timedelta(seconds=duration)
    video_task.status = 'completed'
    video_task.save()

