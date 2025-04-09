from django.shortcuts import render
from .forms import VideoUploadForm
from .models import VideoProcessing
from .utils import process_video_task

def home(request):
    form = VideoUploadForm()
    return render(request,'truesaight/home.html', {'form': form})

def upload_video(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        
        if form.is_valid():
            # 1. Pull the UploaddedFile out of the Form
            video_file = form.cleaned_data['video_file']

            # 2. Make a new instance of the Model for 
            # data initialization into the DB
            video_task = VideoProcessing.objects.create(
                video_file=video_file,
                status='queued',
                confidence_score=0.0,
                processed_frames_count=0
            )
            
            # 3. Call the Service & Change Status to Processing
            process_video_task(video_task)

            return render(request, 'truesaight/result.html', {'video_name': video_file.name})
    else:
        form = VideoUploadForm()

    return render(request, 'truesaight/upload.html', {'form': form})

# def result(request):
#     return render(request, 'truesaight/result.html')
