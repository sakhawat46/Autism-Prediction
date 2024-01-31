from django.shortcuts import render
import keras
from PIL import Image
import numpy as np
import os
from django.core.files.storage import FileSystemStorage

media = 'media'
model = keras.models.load_model('')

# Create your views here.

def home(request):
    return render(request, 'ASD_app/home.html', context={})



def makepredictions(path):
    img = Image.open(path)
    img_d = img.resize((224,224))