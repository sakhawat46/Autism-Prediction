import os
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from PIL import Image
import numpy as np
import keras

media = 'media'

# শুধুমাত্র Multimodal মডেলটি লোড করা হচ্ছে
try:
    multimodal_model = keras.models.load_model('final_model_multimodal_best.keras')
    print("Successfully loaded final_model_multimodal_best.keras")
except Exception as e:
    multimodal_model = None
    print(f"Error loading model: {e}")

prediction_history = []

@csrf_exempt
def predict_api(request):
    if request.method == 'POST':
        try:
            image_files = request.FILES.getlist('files')
            
            if not image_files:
                return JsonResponse({'error': 'No image files provided'}, status=400)
            
            # Age এবং Gender রিসিভ করা (Multimodal এর জন্য বাধ্যতামূলক)
            age = request.POST.get('age')
            gender = request.POST.get('gender')
            
            if not age or not gender:
                return JsonResponse({'error': 'Age and Gender are required'}, status=400)
            
            age = float(age)
            gender = int(gender)
            
            # ফাইল সেভ করা এবং প্রি-প্রসেসিং
            processed_images = []
            fss = FileSystemStorage()
            
            for image_file in image_files:
                file = fss.save(image_file.name, image_file)
                file_path = os.path.join(media, file)
                
                img = Image.open(file_path)
                img_d = img.resize((224, 224))
                if len(np.array(img_d).shape) < 4:
                    rgb_img = Image.new("RGB", img_d.size)
                    rgb_img.paste(img_d)
                else:
                    rgb_img = img_d
                
                rgb_img = np.array(rgb_img, dtype=np.float64)
                processed_images.append(rgb_img)
            
            batch_images = np.array(processed_images)
            
            # Shape error ফিক্স করার জন্য ৬টি ভ্যালুর অ্যারে তৈরি করা হলো
            demographics = np.array([[age, gender, 0, 0, 0, 0]] * len(image_files), dtype=np.float64)
            
            # মডেল প্রেডিকশন এবং TTA Average
            if multimodal_model is not None:
                predictions = multimodal_model.predict([batch_images, demographics], verbose=0)
                final_prediction = np.mean(predictions, axis=0) # রোগীর সব ইমেজের (TTA) গড়
                
                a = int(np.argmax(final_prediction))
                confidence_score = float(np.max(final_prediction))
                confidence = round(confidence_score * 100, 2) if confidence_score <= 1.0 else round(confidence_score, 2)
                diagnosis = "ASD" if a == 1 else "Non-ASD"
            else:
                diagnosis = "Error: Model not loaded"
                confidence = 0.0
            
            record = {
                'diagnosis': diagnosis,
                'confidence': confidence,
                'model': 'Proposed Multimodal + TTA'
            }
            prediction_history.append(record)
            
            return JsonResponse(record)
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
            
    return JsonResponse({'error': 'Invalid request method'}, status=405)


def history_api(request):
    return JsonResponse(prediction_history, safe=False)


def home(request):
    return render(request, 'index.html')
