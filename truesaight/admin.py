from django.contrib import admin
from .models import VideoProcessing

# Register your models here.

class VideoProcessingAdmin(admin.ModelAdmin):
    list_display = ('id', 'video_file', 'processing_reference_number', 'result', 'timestamp', 'status')
    readonly_fields = ('processing_reference_number',)  # Make it read-only

admin.site.register(VideoProcessing, VideoProcessingAdmin)