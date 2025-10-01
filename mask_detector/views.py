import cv2
import numpy as np
from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators import gzip
from django.views.decorators.csrf import csrf_exempt
from keras.models import load_model
import base64
import json

# Load the Keras model
try:
    model = load_model("./model2-010.keras")
    print("Keras model loaded successfully")
except Exception as e:
    print(f"Error loading Keras model: {e}")
    model = None

results = {0: 'without mask', 1: 'mask'}
GR_dict = {0: (0, 0, 255), 1: (0, 255, 0)}

# Load Haar cascade with multiple fallback paths
def load_haar_cascade():
    cascade_paths = [
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
        cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml',
        cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml',
        'haarcascade_frontalface_default.xml',
        './haarcascade_frontalface_default.xml'
    ]
    
    for path in cascade_paths:
        try:
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                print(f"Haar cascade loaded successfully from: {path}")
                return cascade
        except Exception as e:
            print(f"Failed to load from {path}: {e}")
            continue
    
    print("Warning: Using default Haar cascade with potential issues")
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

haarcascade = load_haar_cascade()

def index(request):
    return render(request, 'index.html')

@csrf_exempt
def process_frame(request):
    if request.method == 'POST':
        try:
            if model is None or haarcascade is None:
                return JsonResponse({'error': 'Model or Haar cascade not loaded'}, status=500)
            
            data = json.loads(request.body)
            img_data = data['image']
            
            # Remove data URL prefix if present
            if ',' in img_data:
                img_data = img_data.split(',')[1]
            
            # Decode base64 image
            img_bytes = base64.b64decode(img_data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            im = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if im is None:
                return JsonResponse({'error': 'Failed to decode image'}, status=400)
            
            print(f"Received frame shape: {im.shape}")

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            
            # Apply multiple preprocessing techniques to improve detection
            # 1. Equalize histogram
            gray_eq = cv2.equalizeHist(gray)
            
            # 2. Apply Gaussian blur to reduce noise
            gray_blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)
            
            # Try multiple detection parameters
            faces = []
            
            # Method 1: Standard parameters
            faces1 = haarcascade.detectMultiScale(
                gray_blur,
                scaleFactor=1.05,  # Reduced for better detection
                minNeighbors=3,    # Reduced for more sensitivity
                minSize=(50, 50),  # Increased minimum size
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Method 2: More sensitive parameters
            faces2 = haarcascade.detectMultiScale(
                gray_blur,
                scaleFactor=1.1,
                minNeighbors=2,    # Even more sensitive
                minSize=(30, 30),  # Smaller faces
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Combine results, removing duplicates
            all_faces = list(faces1) + list(faces2)
            
            # Remove duplicate faces (simple IoU-based deduplication)
            final_faces = []
            for (x, y, w, h) in all_faces:
                is_duplicate = False
                for (fx, fy, fw, fh) in final_faces:
                    # Calculate intersection over union
                    xi = max(x, fx)
                    yi = max(y, fy)
                    wi = min(x + w, fx + fw) - xi
                    hi = min(y + h, fy + fh) - yi
                    
                    if wi > 0 and hi > 0:
                        intersection = wi * hi
                        union = w * h + fw * fh - intersection
                        iou = intersection / union if union > 0 else 0
                        
                        if iou > 0.5:  # If overlap is more than 50%, consider duplicate
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    final_faces.append((x, y, w, h))
            
            faces = final_faces
            print(f"Detected {len(faces)} faces after deduplication")
            
            face_data = []
            for (x, y, w, h) in faces:
                try:
                    # Extract face region with boundary checks
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(im.shape[1], x + w), min(im.shape[0], y + h)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                        
                    face_img = im[y1:y2, x1:x2]
                    
                    # Resize to match model input size
                    resized_face = cv2.resize(face_img, (150, 150))
                    
                    # Normalize and reshape for model
                    normalized = resized_face / 255.0
                    reshaped = np.reshape(normalized, (1, 150, 150, 3))
                    
                    # Predict mask/no mask
                    result = model.predict(reshaped, verbose=0)
                    label_idx = np.argmax(result, axis=1)[0]
                    confidence = np.max(result)
                    
                    label = results[label_idx]
                    
                    face_data.append({
                        'x': int(x),
                        'y': int(y),
                        'w': int(w),
                        'h': int(h),
                        'label': label,
                        'confidence': float(confidence)
                    })
                    
                    print(f"Face detected at ({x}, {y}, {w}, {h}) - {label} (confidence: {confidence:.2f})")
                    
                except Exception as face_error:
                    print(f"Error processing face: {face_error}")
                    continue

            return JsonResponse({'faces': face_data})
            
        except Exception as e:
            print(f"Error in process_frame: {str(e)}")
            import traceback
            traceback.print_exc()
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=400)

@gzip.gzip_page
def video_feed(request):
    """Optional video feed endpoint for testing"""
    def generate():
        cap = cv2.VideoCapture(0)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Encode as JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        finally:
            cap.release()
    
    return StreamingHttpResponse(
        generate(), 
        content_type='multipart/x-mixed-replace; boundary=frame'
    )