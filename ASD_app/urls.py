from django.urls import path
from ASD_app import views

app_name = "ASD_app"

urlpatterns = [
    # path('', views.home, name='home'),
    path('', views.index, name='index'),

]