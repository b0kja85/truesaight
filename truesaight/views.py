from django.shortcuts import render
from .forms import VideoUploadForm

def upload_video(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video = form.cleaned_data['video']
            # Save and process video frames here
            return render(request, 'truesaight/result.html', {'video_name': video.name})
    else:
        form = VideoUploadForm()
    return render(request, 'truesaight/upload.html', {'form': form})

def result(request):
    return render(request, 'truesaight/result.html')

def home(request):
    return render(request,'truesaight/home.html')