from django.shortcuts import render
import keras
from PIL import Image
import numpy as np
import os
from django.core.files.storage import FileSystemStorage

media = 'media'
model = keras.models.load_model('')

# Create your views here.

# def home(request):
#     return render(request, 'ASD_app/home.html', context={})



def makepredictions(path):
    img = Image.open(path)
    img_d = img.resize((224,224))

    # We check image RGB or Not
    if len(np.array(img_d).shape)<4:
        rgb_img = Image.new("RGB",img_d.size)
        rgb_img.paste(img_d)
    else:
        rgb_img = img_d

    # Here we conver the image into numpy array and reshape
    rgb_img = np.array(rgb_img, dtype=np.float64)
    rgb_img = rgb_img.reshape(1, 244, 244, 3)

    # We Make Prediction Here
    predictions = model.predict(rgb_img)
    a = int(np.argmax(predictions))
    if a==1:
        a = "Result : Glioma Tumor"
    elif a==2:
        a = "Result : Meningioma Tumor"
    elif a==3:
        a = "Result : No Tumor"
    else:
        a = "Result : Pictiuary Tumor"
    return a




def index(request):
    if request.method =="POST" and request.FILES == ['upload']:

        if 'upload' not in request.FILES:
            err = 'No Images Selected'
            return render(request, 'ASD_app/home.html', {'err': err})
        
        f = request.FILES['upload']
        if f == '':
            err = 'No Files Selected'
            return render(request, 'ASD_app/home.html', {'err': err})
        
        upload = request.FILES['upload']
        fss = FileSystemStorage()
        file = fss.save(upload.name, upload)
        file_url = fss.url(file)
        predictions = makepredictions(os.path.join(media, file))
        return render(request, 'ASD_app/home.html', {'pred': predictions, 'file_url': file_url})
    else:
        return render(request, 'ASD_app/home.html')