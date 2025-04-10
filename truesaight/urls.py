from django.urls import path
from . import views

app_name = 'truesaight'

urlpatterns = [
    path('', views.home, name='home'),  # Homepage with Upload Form
    path('upload/', views.upload_video, name='upload_video'),  # Handle Upload POST
    path('processing/<int:pk>/', views.processing_view, name='processing'),  # Show loading spinner
    path('result/<int:pk>/', views.result_view, name='result'),  # Final results
]

# Serving the Media Files

from django.conf import settings
from django.conf.urls.static import static

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)