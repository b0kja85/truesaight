import time
from datetime import timedelta
from .ml_utils import run_deepfake_model

def process_video_task(video_task): 
    """
    Given a VideoProcessing instance (with .video_file.path),
    extract frames, run the ML model, and update the instance.
    """
    # mark as processing
    video_task.status = 'processing'
    video_task.save(update_fields=['status'])

    start = time.time()

    # run your model, returns (result, confidence, frames_count)
    result, confidence, frames_count = run_deepfake_model(video_task.video_file.path)

    duration = time.time() - start

    # update the record
    video_task.result = result
    video_task.confidence_score = confidence
    video_task.processed_frames_count = frames_count
    video_task.processing_duration = timedelta(seconds=duration)
    video_task.status = 'completed'
    video_task.save()
