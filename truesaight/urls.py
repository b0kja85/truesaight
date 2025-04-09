from django.urls import path
from . import views

app_name = 'truesaight'

urlpatterns = [
    path('', views.home, name='home'), # Default Home    
    path('upload/', views.upload_video, name='upload_video')
    # path('results/<int:pk>', views.result, name='result')

]