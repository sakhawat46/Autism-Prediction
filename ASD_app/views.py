import os
import cv2
import numpy as np
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
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

# ==========================================
# রিসার্চ পেপারের এক্স্যাক্ট প্রি-প্রসেসিং এবং ফিচার এক্সট্রাকশন
# ==========================================
def preprocess_sharpened(img_bgr, size=224):
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    dilated = cv2.dilate(img_bgr, kernel_dilate, iterations=1)
    
    SHARPEN_KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(dilated.astype(np.float32), -1, SHARPEN_KERNEL)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    img_resized = cv2.resize(sharpened, (size, size), interpolation=cv2.INTER_AREA)
    img_resized = cv2.filter2D(img_resized.astype(np.float32), -1, SHARPEN_KERNEL)
    img_resized = np.clip(img_resized, 0, 255).astype(np.uint8)
    
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_resized[gray < 3] = 0
    
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    return img_rgb.astype(np.float32) / 255.0

def compute_scanpath_density(img):
    gray = np.mean(img, axis=-1)
    return float(np.mean(gray > 0.05) * 100.0)

# ==========================================
# API Endpoint
# ==========================================
@csrf_exempt
def predict_api(request):
    if request.method == 'POST':
        try:
            image_files = request.FILES.getlist('files')
            if not image_files:
                return JsonResponse({'error': 'No image files provided'}, status=400)
            
            age = request.POST.get('age')
            gender = request.POST.get('gender')
            if not age or not gender:
                return JsonResponse({'error': 'Age and Gender are required'}, status=400)
            
            age_num = float(age)
            gender_encoded = float(gender)
            
            processed_images = []
            meta_features = []
            fss = FileSystemStorage()
            
            age_mean = 8.2
            age_std = 2.8
            
            for image_file in image_files:
                file = fss.save(image_file.name, image_file)
                file_path = os.path.join(media, file)
                
                # ইমেজ রিড করা
                img_bgr = cv2.imread(file_path)
                if img_bgr is None:
                    continue
                
                # নোটবুকের মতো হুবহু শার্পেনিং এবং রিসাইজিং (224x224)
                img_norm = preprocess_sharpened(img_bgr, size=224)
                processed_images.append(img_norm)
                
                # ৬টি মেটাডেটা ফিচার ক্যালকুলেশন
                age_norm_val = (age_num - age_mean) / (age_std + 1e-5)
                dens = compute_scanpath_density(img_norm)
                
                gray_img = np.mean(img_norm, axis=-1)
                h, w = gray_img.shape
                center_crop = gray_img[h//4 : 3*h//4, w//4 : 3*w//4]
                center_ratio = float(np.mean(center_crop > 0.05) / (np.mean(gray_img > 0.05) + 1e-5))
                
                ys, xs = np.where(gray_img > 0.05)
                spatial_spread = float((np.std(ys)/h + np.std(xs)/w)) if len(ys) > 5 else 0.0
                rg_diff = float(np.mean(img_norm[:, :, 0]) - np.mean(img_norm[:, :, 1])) * 10.0
                
                # ৬টি কলামের ডেমোগ্রাফিক্স অ্যারে
                meta_features.append([age_norm_val, gender_encoded, dens / 5.0, center_ratio, spatial_spread, rg_diff])
            
            if len(processed_images) == 0:
                return JsonResponse({'error': 'Failed to process images.'}, status=500)

            batch_images = np.array(processed_images, dtype=np.float32)
            demographics = np.array(meta_features, dtype=np.float32)
            
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
