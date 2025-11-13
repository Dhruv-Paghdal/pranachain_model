from django.urls import path
from . import views

urlpatterns = [
    path('', views.ckd_interface, name='ckd_interface'),
    path('predict-file/', views.predict_ckd_file, name='predict_ckd_file'),
]