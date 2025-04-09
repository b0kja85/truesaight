from django.contrib import admin
from .models import VideoProcessing

# Register your models here.

class VideoProcessingAdmin(admin.ModelAdmin):
    list_display = ('id', 'video_file', 'result', 'timestamp', 'status')

admin.site.register(VideoProcessing, VideoProcessingAdmin)