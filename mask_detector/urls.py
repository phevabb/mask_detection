from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('process_frame/', views.process_frame, name='process_frame'),
    path('video_feed/', views.video_feed, name='video_feed'),
]