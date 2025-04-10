from django.shortcuts import render, redirect, get_object_or_404
from .forms import VideoUploadForm
from .models import VideoProcessing
from .utils import process_video_task
from threading import Thread

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

            # Kick off async background processing
            Thread(target=process_video_task, args=(video_task.id,)).start()

            # Redirect to spinner while processing
            return redirect('truesaight:processing', pk=video_task.id)
    else:
        form = VideoUploadForm()

    return render(request, 'truesaight/upload.html', {'form': form})

def processing_view(request, pk):
    video = get_object_or_404(VideoProcessing, pk=pk)

    # Redirect to result when processing is done
    if video.status == 'completed':
        return redirect('truesaight:result', pk=video.id)

    return render(request, 'truesaight/processing.html', {'video': video})

def result_view(request, pk):
    video = get_object_or_404(VideoProcessing, pk=pk)
    return render(request, 'truesaight/result.html', {'video': video})
