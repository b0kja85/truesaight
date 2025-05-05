from django.shortcuts import render, redirect, get_object_or_404
from .forms import VideoUploadForm
from .models import VideoProcessing
from .utils.app_utils import process_video_task
from threading import Thread
from django.http import JsonResponse
from django.conf import settings
import os

def home(request):
    form = VideoUploadForm()
    return render(request, 'truesaight/home.html', {'form': form})

def upload_video(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        
        if form.is_valid():
            video_file = form.cleaned_data['video_file']

            video_task = VideoProcessing.objects.create(
                video_file=video_file,
                status='queued',
                confidence_score=0.0,
                processed_frames_count=0
            )

            # Redirect to spinner while processing
            return redirect('truesaight:processing', pk=video_task.id)
    else:
        form = VideoUploadForm()

    return render(request, 'truesaight/upload.html', {'form': form})

def processing_view(request, pk):
    video = get_object_or_404(VideoProcessing, pk=pk)

    # Kick off async background processing if not already started
    if video.status != 'completed' and video.status != 'failed':
        Thread(target=process_video_task, args=(video.id,)).start()

    # If video status is 'completed', redirect to result page
    if video.status == 'completed':
        return redirect('truesaight:result', pk=video.id)

    return render(request, 'truesaight/processing.html', {'video': video})

def get_video_status(request, pk):
    video = get_object_or_404(VideoProcessing, pk=pk)
    return JsonResponse({'status': video.status})

def result_view(request, pk):
    video = get_object_or_404(VideoProcessing, pk=pk)
    frame_dir = os.path.join(settings.MEDIA_ROOT, 'extracted_frames', str(pk))
    frames = sorted(os.listdir(frame_dir)) if os.path.exists(frame_dir) else []

    # Scale the confidence to percentage for display
    confidence_percentage = video.confidence_score * 100
    fake_percentage = 100 - confidence_percentage

    return render(request, 'truesaight/result.html', {
        'video': video,
        'frame_filenames': frames,
        'confidence_percentage': confidence_percentage,
        'fake_percentage': fake_percentage,
        'MEDIA_URL': settings.MEDIA_URL,
    })