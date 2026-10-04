from django.urls import path
from ASD_app import views

app_name = "ASD_app"

urlpatterns = [
    path('', views.home, name='home'),
    path('predict', views.predict_api, name='predict'),
    path('history', views.history_api, name='history'),
]