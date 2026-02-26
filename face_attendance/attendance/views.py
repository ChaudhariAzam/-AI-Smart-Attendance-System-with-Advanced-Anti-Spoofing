# # from django.shortcuts import render
# # from django.views.decorators.csrf import csrf_exempt
# # from django.utils.decorators import method_decorator
# # from rest_framework.views import APIView
# # from rest_framework.response import Response
# # from rest_framework import status
# # from rest_framework.authentication import SessionAuthentication, BasicAuthentication
# # from .models import User, Attendance
# # import numpy as np
# # import pickle
# # import cv2
# # from insightface.app import FaceAnalysis
# # import io

# # # Initialize InsightFace
# # app = FaceAnalysis(providers=['CPUExecutionProvider'])
# # app.prepare(ctx_id=0, det_size=(640, 640))

# # def embedding_to_binary(embedding):
# #     return pickle.dumps(embedding)

# # def binary_to_embedding(binary):
# #     return pickle.loads(binary)

# # def load_image_to_rgb_filelike(file_obj):
# #     try:
# #         file_obj.seek(0)
# #         image_bytes = file_obj.read()
# #         nparr = np.frombuffer(image_bytes, np.uint8)
# #         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
# #         if img is None:
# #             raise ValueError("Failed to decode image")
# #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #         return img
# #     except Exception as e:
# #         raise ValueError(f"Cannot process image: {e}")

# # def get_face_analysis(img):
# #     try:
# #         faces = app.get(img)
# #         return faces
# #     except Exception as e:
# #         print(f"InsightFace error: {e}")
# #         return []

# # def face_laplacian_variance(img, bbox):
# #     x1, y1, x2, y2 = bbox
# #     h, w = img.shape[:2]
# #     x1 = max(0, int(x1)); y1 = max(0, int(y1))
# #     x2 = min(w - 1, int(x2)); y2 = min(h - 1, int(y2))
# #     if x2 <= x1 or y2 <= y1:
# #         return 0.0
# #     face_region = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
# #     if face_region.size == 0:
# #         return 0.0
# #     return float(cv2.Laplacian(face_region, cv2.CV_64F).var())

# # def extract_landmark_array(face):
# #     lm = getattr(face, 'landmark', None)
# #     if lm is None:
# #         return None
# #     arr = np.array(lm).reshape(-1, 2)
# #     return arr

# # def landmarks_motion_score(landmark_seq):
# #     stacked = np.stack(landmark_seq, axis=0)
# #     std_per_point = np.std(stacked, axis=0)
# #     return float(std_per_point.mean())

# # def detect_liveness(frames):
# #     frames_with_face = [f for f in frames if len(f['faces']) > 0]
# #     if len(frames_with_face) < 2:
# #         return False, "No face consistently detected in multiple frames"

# #     face_entries = []
# #     for f in frames_with_face:
# #         best_face = None
# #         best_area = -1
# #         for face in f['faces']:
# #             bbox = getattr(face, 'bbox', None)
# #             if bbox is None:
# #                 best_face = face
# #                 break
# #             x1, y1, x2, y2 = bbox
# #             area = max(0, (x2 - x1) * (y2 - y1))
# #             if area > best_area:
# #                 best_area = area
# #                 best_face = face
# #         if best_face is not None:
# #             face_entries.append((f['img'], best_face))

# #     if len(face_entries) < 2:
# #         return False, "Face detection unstable across frames"

# #     lap_vars = []
# #     landmarks_list = []
# #     for img, face in face_entries:
# #         bbox = getattr(face, 'bbox', None)
# #         if bbox is None:
# #             h, w = img.shape[:2]
# #             bbox = (0, 0, w-1, h-1)
# #         lv = face_laplacian_variance(img, bbox)
# #         lap_vars.append(lv)
# #         lm = extract_landmark_array(face)
# #         if lm is not None:
# #             landmarks_list.append(lm)

# #     avg_lap = float(np.mean(lap_vars)) if lap_vars else 0.0
# #     if avg_lap < 40.0:
# #         return False, f"Face appears too smooth/low-texture (lap={avg_lap:.1f}) — possible spoof"

# #     if len(landmarks_list) >= 2:
# #         score = landmarks_motion_score(landmarks_list)
# #         if score < 1.5:
# #             return False, f"Very little landmark motion (motion_score={score:.2f}) — possible static photo"
# #     else:
# #         centers = []
# #         for img, face in face_entries:
# #             bbox = getattr(face, 'bbox', None)
# #             if bbox is None:
# #                 continue
# #             x1, y1, x2, y2 = bbox
# #             centers.append(((x1+x2)/2.0, (y1+y2)/2.0))
# #         if len(centers) >= 2:
# #             centers = np.array(centers)
# #             stdc = float(np.std(centers, axis=0).mean())
# #             if stdc < 1.5:
# #                 return False, f"Very little face center motion (center_std={stdc:.2f}) — possible static photo"

# #     return True, "Liveness checks passed"

# # def get_embedding_from_frame(img):
# #     faces = get_face_analysis(img)
# #     if not faces:
# #         return None
# #     return faces[0].embedding

# # # Custom authentication to bypass CSRF
# # class CsrfExemptSessionAuthentication(SessionAuthentication):
# #     def enforce_csrf(self, request):
# #         return  # bypass CSRF check

# # # ------------------- API Views -------------------

# # @method_decorator(csrf_exempt, name='dispatch')
# # class UserRegisterView(APIView):
# #     authentication_classes = (CsrfExemptSessionAuthentication, BasicAuthentication)

# #     def post(self, request):
# #         user_id = request.data.get('user_id')
# #         name = request.data.get('name')
# #         files = request.FILES.getlist('face_image')

# #         if not all([user_id, name]) or not files:
# #             return Response({'error': 'All fields required and at least one frame'}, status=status.HTTP_400_BAD_REQUEST)

# #         frames = []
# #         for f in files:
# #             try:
# #                 img = load_image_to_rgb_filelike(f)
# #             except ValueError as e:
# #                 return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
# #             faces = get_face_analysis(img)
# #             frames.append({'img': img, 'faces': faces})

# #         live, reason = detect_liveness(frames)
# #         if not live:
# #             return Response({'error': 'Liveness check failed', 'detail': reason}, status=status.HTTP_400_BAD_REQUEST)

# #         embedding = None
# #         for fr in frames:
# #             if fr['faces']:
# #                 embedding = fr['faces'][0].embedding
# #                 break

# #         if embedding is None:
# #             return Response({'error': 'No face found for embedding'}, status=status.HTTP_400_BAD_REQUEST)

# #         embedding_binary = embedding_to_binary(embedding)
# #         user, created = User.objects.get_or_create(
# #             user_id=user_id,
# #             defaults={'name': name, 'embedding': embedding_binary}
# #         )
# #         if not created:
# #             user.name = name
# #             user.embedding = embedding_binary
# #             user.save()

# #         return Response({'message': 'User registered successfully'}, status=status.HTTP_201_CREATED)


# # @method_decorator(csrf_exempt, name='dispatch')
# # class PunchAttendanceView(APIView):
# #     authentication_classes = (CsrfExemptSessionAuthentication, BasicAuthentication)

# #     def post(self, request):
# #         files = request.FILES.getlist('face_image')
# #         if not files:
# #             return Response({'error': 'Face image(s) required'}, status=status.HTTP_400_BAD_REQUEST)

# #         frames = []
# #         for f in files:
# #             try:
# #                 img = load_image_to_rgb_filelike(f)
# #             except ValueError as e:
# #                 return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
# #             faces = get_face_analysis(img)
# #             frames.append({'img': img, 'faces': faces})

# #         live, reason = detect_liveness(frames)
# #         if not live:
# #             return Response({'error': 'Liveness check failed', 'detail': reason}, status=status.HTTP_400_BAD_REQUEST)

# #         new_embedding = None
# #         for fr in frames:
# #             if fr['faces']:
# #                 new_embedding = fr['faces'][0].embedding
# #                 break

# #         if new_embedding is None:
# #             return Response({'error': 'No face detected'}, status=status.HTTP_400_BAD_REQUEST)

# #         users = User.objects.all()
# #         best_match = None
# #         highest_similarity = 0.0

# #         for user in users:
# #             try:
# #                 stored_embedding = binary_to_embedding(user.embedding)
# #             except Exception:
# #                 continue
# #             sim = float(np.dot(stored_embedding, new_embedding) / (np.linalg.norm(stored_embedding) * np.linalg.norm(new_embedding)))
# #             if sim > highest_similarity and sim > 0.4:
# #                 highest_similarity = sim
# #                 best_match = user

# #         if best_match:
# #             Attendance.objects.create(user=best_match)
# #             return Response({
# #                 'message': f'Attendance recorded for {best_match.name}',
# #                 'confidence': f'{highest_similarity:.2%}'
# #             }, status=status.HTTP_200_OK)

# #         return Response({'error': 'User not recognized'}, status=status.HTTP_404_NOT_FOUND)


# # # Page render (no CSRF issue for GET)
# # def attendance_page(request):
# #     return render(request, 'attendance.html')
# from django.shortcuts import render
# from django.views.decorators.csrf import csrf_exempt
# from django.utils.decorators import method_decorator
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# from rest_framework.authentication import SessionAuthentication, BasicAuthentication
# from .models import User, Attendance
# import numpy as np
# import pickle
# import cv2
# from insightface.app import FaceAnalysis
# import io

# # Initialize InsightFace
# app = FaceAnalysis(providers=['CPUExecutionProvider'])
# app.prepare(ctx_id=0, det_size=(640, 640))

# def embedding_to_binary(embedding):
#     return pickle.dumps(embedding)

# def binary_to_embedding(binary):
#     return pickle.loads(binary)

# def load_image_to_rgb_filelike(file_obj):
#     try:
#         file_obj.seek(0)
#         image_bytes = file_obj.read()
#         nparr = np.frombuffer(image_bytes, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         if img is None:
#             raise ValueError("Failed to decode image")
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         return img
#     except Exception as e:
#         raise ValueError(f"Cannot process image: {e}")

# def get_face_analysis(img):
#     try:
#         faces = app.get(img)
#         return faces
#     except Exception as e:
#         print(f"InsightFace error: {e}")
#         return []

# def detect_screen_patterns(img, bbox):
#     """Detect screen moiré patterns and digital display artifacts"""
#     x1, y1, x2, y2 = bbox
#     h, w = img.shape[:2]
#     x1 = max(0, int(x1)); y1 = max(0, int(y1))
#     x2 = min(w - 1, int(x2)); y2 = min(h - 1, int(y2))
#     if x2 <= x1 or y2 <= y1:
#         return 0.0, 0.0
    
#     face_region = img[y1:y2, x1:x2]
    
#     # Resize for faster processing
#     if face_region.shape[0] > 128 or face_region.shape[1] > 128:
#         face_region = cv2.resize(face_region, (128, 128), interpolation=cv2.INTER_AREA)
    
#     gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
    
#     # FFT analysis for periodic patterns (moiré effect from screens)
#     f = np.fft.fft2(gray)
#     fshift = np.fft.fftshift(f)
#     magnitude_spectrum = np.abs(fshift)
    
#     # Detect high-frequency periodic patterns typical of screens
#     rows, cols = magnitude_spectrum.shape
#     crow, ccol = rows // 2, cols // 2
    
#     # Create mask to isolate mid-high frequencies
#     mask = np.ones((rows, cols), np.uint8)
#     r_inner = 10
#     r_outer = min(rows, cols) // 3
#     y, x = np.ogrid[:rows, :cols]
#     mask_area = (x - ccol)**2 + (y - crow)**2
#     mask[(mask_area <= r_inner**2) | (mask_area >= r_outer**2)] = 0
    
#     # Calculate energy in mid-frequency band (screen patterns)
#     mid_freq_energy = np.sum(magnitude_spectrum * mask) / np.sum(mask)
#     total_energy = np.sum(magnitude_spectrum) / (rows * cols)
#     freq_ratio = mid_freq_energy / (total_energy + 1e-10)
    
#     # Color histogram analysis - screens have more uniform color distribution
#     hsv = cv2.cvtColor(face_region, cv2.COLOR_RGB2HSV)
#     hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
#     hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    
#     # Normalize histograms
#     hist_h = hist_h.flatten() / (hist_h.sum() + 1e-10)
#     hist_s = hist_s.flatten() / (hist_s.sum() + 1e-10)
    
#     # Calculate entropy (lower entropy = more uniform = possible screen)
#     entropy_h = -np.sum(hist_h * np.log2(hist_h + 1e-10))
#     entropy_s = -np.sum(hist_s * np.log2(hist_s + 1e-10))
#     avg_entropy = (entropy_h + entropy_s) / 2
    
#     return freq_ratio, avg_entropy

# def face_texture_analysis(img, bbox):
#     """Analyze face texture to detect printed photos or screens"""
#     x1, y1, x2, y2 = bbox
#     h, w = img.shape[:2]
#     x1 = max(0, int(x1)); y1 = max(0, int(y1))
#     x2 = min(w - 1, int(x2)); y2 = min(h - 1, int(y2))
#     if x2 <= x1 or y2 <= y1:
#         return 0.0, 0.0
    
#     face_region = img[y1:y2, x1:x2]
    
#     # Resize for consistent processing
#     if face_region.shape[0] > 128 or face_region.shape[1] > 128:
#         face_region = cv2.resize(face_region, (128, 128), interpolation=cv2.INTER_AREA)
    
#     gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
    
#     # Laplacian variance - measures texture sharpness
#     laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
#     # Gradient magnitude - real faces have varied depth
#     sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
#     sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
#     gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
#     gradient_std = float(np.std(gradient_magnitude))
    
#     return laplacian_var, gradient_std

# def detect_reflections_and_glare(img, bbox):
#     """Detect screen reflections and glare patterns"""
#     x1, y1, x2, y2 = bbox
#     h, w = img.shape[:2]
#     x1 = max(0, int(x1)); y1 = max(0, int(y1))
#     x2 = min(w - 1, int(x2)); y2 = min(h - 1, int(y2))
#     if x2 <= x1 or y2 <= y1:
#         return 0.0
    
#     face_region = img[y1:y2, x1:x2]
#     gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
    
#     # Detect bright spots (screen glare/reflections)
#     threshold_value = np.percentile(gray, 95)
#     bright_mask = gray > threshold_value
#     bright_ratio = np.sum(bright_mask) / gray.size
    
#     return float(bright_ratio)

# def detect_anti_spoofing(img):
#     """Single image anti-spoofing check - detects screens and printed photos"""
    
#     faces = get_face_analysis(img)
#     if not faces:
#         return False, "No face detected"
    
#     # Get the largest face
#     best_face = None
#     best_area = -1
#     for face in faces:
#         bbox = getattr(face, 'bbox', None)
#         if bbox is None:
#             best_face = face
#             break
#         x1, y1, x2, y2 = bbox
#         area = max(0, (x2 - x1) * (y2 - y1))
#         if area > best_area:
#             best_area = area
#             best_face = face
    
#     if best_face is None:
#         return False, "No valid face detected"
    
#     bbox = getattr(best_face, 'bbox', None)
#     if bbox is None:
#         h, w = img.shape[:2]
#         bbox = (0, 0, w-1, h-1)
    
#     # 1. Screen Pattern Detection
#     freq_ratio, color_entropy = detect_screen_patterns(img, bbox)
    
#     if freq_ratio > 1.8:
#         return False, f"Digital display patterns detected (freq_ratio={freq_ratio:.2f})"
    
#     if color_entropy < 3.8:
#         return False, f"Screen-like color uniformity detected (entropy={color_entropy:.2f})"
    
#     # 2. Texture Analysis
#     laplacian_var, gradient_std = face_texture_analysis(img, bbox)
    
#     if laplacian_var < 50.0:
#         return False, f"Low texture quality detected (lap={laplacian_var:.1f}) — possible photo/screen"
    
#     if gradient_std < 8.0:
#         return False, f"Flat surface detected (gradient={gradient_std:.2f}) — possible 2D image"
    
#     # 3. Reflection/Glare Detection
#     bright_ratio = detect_reflections_and_glare(img, bbox)
    
#     if bright_ratio > 0.15:
#         return False, f"Excessive glare detected (bright_ratio={bright_ratio:.2f}) — possible screen reflection"
    
#     return True, "Anti-spoofing checks passed"

# def get_embedding_from_frame(img):
#     faces = get_face_analysis(img)
#     if not faces:
#         return None
#     return faces[0].embedding

# # Custom authentication to bypass CSRF
# class CsrfExemptSessionAuthentication(SessionAuthentication):
#     def enforce_csrf(self, request):
#         return  # bypass CSRF check

# # ------------------- API Views -------------------

# @method_decorator(csrf_exempt, name='dispatch')
# class UserRegisterView(APIView):
#     authentication_classes = (CsrfExemptSessionAuthentication, BasicAuthentication)

#     def post(self, request):
#         user_id = request.data.get('user_id')
#         name = request.data.get('name')
#         face_image = request.FILES.get('face_image')

#         if not all([user_id, name, face_image]):
#             return Response({'error': 'All fields required'}, status=status.HTTP_400_BAD_REQUEST)

#         try:
#             img = load_image_to_rgb_filelike(face_image)
#         except ValueError as e:
#             return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

#         # Anti-spoofing check
#         is_real, reason = detect_anti_spoofing(img)
#         if not is_real:
#             return Response({'error': 'Anti-spoofing check failed', 'detail': reason}, status=status.HTTP_400_BAD_REQUEST)

#         # Get face embedding
#         faces = get_face_analysis(img)
#         if not faces:
#             return Response({'error': 'No face detected'}, status=status.HTTP_400_BAD_REQUEST)

#         embedding = faces[0].embedding
#         embedding_binary = embedding_to_binary(embedding)
        
#         user, created = User.objects.get_or_create(
#             user_id=user_id,
#             defaults={'name': name, 'embedding': embedding_binary}
#         )
#         if not created:
#             user.name = name
#             user.embedding = embedding_binary
#             user.save()

#         return Response({'message': 'User registered successfully'}, status=status.HTTP_201_CREATED)


# @method_decorator(csrf_exempt, name='dispatch')
# class PunchAttendanceView(APIView):
#     authentication_classes = (CsrfExemptSessionAuthentication, BasicAuthentication)

#     def post(self, request):
#         face_image = request.FILES.get('face_image')
#         if not face_image:
#             return Response({'error': 'Face image required'}, status=status.HTTP_400_BAD_REQUEST)

#         try:
#             img = load_image_to_rgb_filelike(face_image)
#         except ValueError as e:
#             return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

#         # Anti-spoofing check
#         is_real, reason = detect_anti_spoofing(img)
#         if not is_real:
#             return Response({'error': 'Anti-spoofing check failed', 'detail': reason}, status=status.HTTP_400_BAD_REQUEST)

#         # Get face embedding
#         faces = get_face_analysis(img)
#         if not faces:
#             return Response({'error': 'No face detected'}, status=status.HTTP_400_BAD_REQUEST)

#         new_embedding = faces[0].embedding

#         # Match with stored users
#         users = User.objects.all()
#         best_match = None
#         highest_similarity = 0.0

#         for user in users:
#             try:
#                 stored_embedding = binary_to_embedding(user.embedding)
#             except Exception:
#                 continue
#             sim = float(np.dot(stored_embedding, new_embedding) / (np.linalg.norm(stored_embedding) * np.linalg.norm(new_embedding)))
#             if sim > highest_similarity and sim > 0.4:
#                 highest_similarity = sim
#                 best_match = user

#         if best_match:
#             Attendance.objects.create(user=best_match)
#             return Response({
#                 'message': f'Attendance recorded for {best_match.name}',
#                 'confidence': f'{highest_similarity:.2%}'
#             }, status=status.HTTP_200_OK)

#         return Response({'error': 'User not recognized'}, status=status.HTTP_404_NOT_FOUND)


# # Page render (no CSRF issue for GET)
# def attendance_page(request):
#     return render(request, 'attendance.html')
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.authentication import SessionAuthentication, BasicAuthentication
from .models import User, Attendance
import numpy as np
import pickle
import cv2
from insightface.app import FaceAnalysis
import io

# Initialize InsightFace
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def embedding_to_binary(embedding):
    return pickle.dumps(embedding)

def binary_to_embedding(binary):
    return pickle.loads(binary)

def load_image_to_rgb_filelike(file_obj):
    try:
        file_obj.seek(0)
        image_bytes = file_obj.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        raise ValueError(f"Cannot process image: {e}")

def get_face_analysis(img):
    try:
        faces = app.get(img)
        return faces
    except Exception as e:
        print(f"InsightFace error: {e}")
        return []

def detect_screen_patterns(img, bbox):
    """Detect screen moiré patterns and digital display artifacts"""
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w - 1, int(x2)); y2 = min(h - 1, int(y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0
    
    face_region = img[y1:y2, x1:x2]
    
    # Resize for faster processing
    if face_region.shape[0] > 128 or face_region.shape[1] > 128:
        face_region = cv2.resize(face_region, (128, 128), interpolation=cv2.INTER_AREA)
    
    gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
    
    # FFT analysis for periodic patterns (moiré effect from screens)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    
    # Detect high-frequency periodic patterns typical of screens
    rows, cols = magnitude_spectrum.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create mask to isolate mid-high frequencies
    mask = np.ones((rows, cols), np.uint8)
    r_inner = 10
    r_outer = min(rows, cols) // 3
    y, x = np.ogrid[:rows, :cols]
    mask_area = (x - ccol)**2 + (y - crow)**2
    mask[(mask_area <= r_inner**2) | (mask_area >= r_outer**2)] = 0
    
    # Calculate energy in mid-frequency band (screen patterns)
    mid_freq_energy = np.sum(magnitude_spectrum * mask) / np.sum(mask)
    total_energy = np.sum(magnitude_spectrum) / (rows * cols)
    freq_ratio = mid_freq_energy / (total_energy + 1e-10)
    
    # Color histogram analysis - screens have more uniform color distribution
    hsv = cv2.cvtColor(face_region, cv2.COLOR_RGB2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    
    # Normalize histograms
    hist_h = hist_h.flatten() / (hist_h.sum() + 1e-10)
    hist_s = hist_s.flatten() / (hist_s.sum() + 1e-10)
    
    # Calculate entropy (lower entropy = more uniform = possible screen)
    entropy_h = -np.sum(hist_h * np.log2(hist_h + 1e-10))
    entropy_s = -np.sum(hist_s * np.log2(hist_s + 1e-10))
    avg_entropy = (entropy_h + entropy_s) / 2
    
    return freq_ratio, avg_entropy

def face_texture_analysis(img, bbox):
    """Analyze face texture to detect printed photos or screens"""
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w - 1, int(x2)); y2 = min(h - 1, int(y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0
    
    face_region = img[y1:y2, x1:x2]
    
    # Resize for consistent processing
    if face_region.shape[0] > 128 or face_region.shape[1] > 128:
        face_region = cv2.resize(face_region, (128, 128), interpolation=cv2.INTER_AREA)
    
    gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
    
    # Laplacian variance - measures texture sharpness
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    # Gradient magnitude - real faces have varied depth
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_std = float(np.std(gradient_magnitude))
    
    return laplacian_var, gradient_std

def detect_reflections_and_glare(img, bbox):
    """Detect screen reflections and glare patterns"""
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w - 1, int(x2)); y2 = min(h - 1, int(y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    face_region = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
    
    # Detect bright spots (screen glare/reflections)
    threshold_value = np.percentile(gray, 95)
    bright_mask = gray > threshold_value
    bright_ratio = np.sum(bright_mask) / gray.size
    
    return float(bright_ratio)

def detect_anti_spoofing(img):
    """Single image anti-spoofing check - detects screens and printed photos"""
    
    faces = get_face_analysis(img)
    if not faces:
        return False, "No face detected"
    
    # Get the largest face
    best_face = None
    best_area = -1
    for face in faces:
        bbox = getattr(face, 'bbox', None)
        if bbox is None:
            best_face = face
            break
        x1, y1, x2, y2 = bbox
        area = max(0, (x2 - x1) * (y2 - y1))
        if area > best_area:
            best_area = area
            best_face = face
    
    if best_face is None:
        return False, "No valid face detected"
    
    bbox = getattr(best_face, 'bbox', None)
    if bbox is None:
        h, w = img.shape[:2]
        bbox = (0, 0, w-1, h-1)
    
    # 1. Screen Pattern Detection
    freq_ratio, color_entropy = detect_screen_patterns(img, bbox)
    
    if freq_ratio > 1.8:
        return False, f"Digital display patterns detected (freq_ratio={freq_ratio:.2f})"
    
    if color_entropy < 3.8:
        return False, f"Screen-like color uniformity detected (entropy={color_entropy:.2f})"
    
    # 2. Texture Analysis
    laplacian_var, gradient_std = face_texture_analysis(img, bbox)
    
    if laplacian_var < 50.0:
        return False, f"Low texture quality detected (lap={laplacian_var:.1f}) — possible photo/screen"
    
    if gradient_std < 8.0:
        return False, f"Flat surface detected (gradient={gradient_std:.2f}) — possible 2D image"
    
    # 3. Reflection/Glare Detection
    bright_ratio = detect_reflections_and_glare(img, bbox)
    
    if bright_ratio > 0.15:
        return False, f"Excessive glare detected (bright_ratio={bright_ratio:.2f}) — possible screen reflection"
    
    return True, "Anti-spoofing checks passed"

def get_embedding_from_frame(img):
    faces = get_face_analysis(img)
    if not faces:
        return None
    return faces[0].embedding

# Custom authentication to bypass CSRF
class CsrfExemptSessionAuthentication(SessionAuthentication):
    def enforce_csrf(self, request):
        return  # bypass CSRF check

# ------------------- API Views -------------------

@method_decorator(csrf_exempt, name='dispatch')
class UserRegisterView(APIView):
    authentication_classes = (CsrfExemptSessionAuthentication, BasicAuthentication)

    def post(self, request):
        user_id = request.data.get('user_id')
        name = request.data.get('name')
        face_image = request.FILES.get('face_image')

        if not all([user_id, name, face_image]):
            return Response({'error': 'All fields required'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            img = load_image_to_rgb_filelike(face_image)
        except ValueError as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

        # Anti-spoofing check
        is_real, reason = detect_anti_spoofing(img)
        if not is_real:
            return Response({'error': 'Anti-spoofing check failed', 'detail': reason}, status=status.HTTP_400_BAD_REQUEST)

        # Get face embedding
        faces = get_face_analysis(img)
        if not faces:
            return Response({'error': 'No face detected'}, status=status.HTTP_400_BAD_REQUEST)

        embedding = faces[0].embedding
        embedding_binary = embedding_to_binary(embedding)
        
        user, created = User.objects.get_or_create(
            user_id=user_id,
            defaults={'name': name, 'embedding': embedding_binary}
        )
        if not created:
            user.name = name
            user.embedding = embedding_binary
            user.save()

        return Response({'message': 'User registered successfully'}, status=status.HTTP_201_CREATED)


@method_decorator(csrf_exempt, name='dispatch')
class PunchAttendanceView(APIView):
    authentication_classes = (CsrfExemptSessionAuthentication, BasicAuthentication)

    def post(self, request):
        face_image = request.FILES.get('face_image')
        if not face_image:
            return Response({'error': 'Face image required'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            img = load_image_to_rgb_filelike(face_image)
        except ValueError as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

        # Anti-spoofing check
        is_real, reason = detect_anti_spoofing(img)
        if not is_real:
            return Response({'error': 'Anti-spoofing check failed', 'detail': reason}, status=status.HTTP_400_BAD_REQUEST)

        # Get face embedding
        faces = get_face_analysis(img)
        if not faces:
            return Response({'error': 'No face detected'}, status=status.HTTP_400_BAD_REQUEST)

        new_embedding = faces[0].embedding

        # Match with stored users
        users = User.objects.all()
        best_match = None
        highest_similarity = 0.0

        for user in users:
            try:
                stored_embedding = binary_to_embedding(user.embedding)
            except Exception:
                continue
            sim = float(np.dot(stored_embedding, new_embedding) / (np.linalg.norm(stored_embedding) * np.linalg.norm(new_embedding)))
            if sim > highest_similarity and sim > 0.4:
                highest_similarity = sim
                best_match = user

        if best_match:
            Attendance.objects.create(user=best_match)
            return Response({
                'message': f'Attendance recorded for {best_match.name}',
                'confidence': f'{highest_similarity:.2%}',
                'user_id': best_match.user_id,   # return user_id
                'name': f'{best_match.name}',
                
            }, status=status.HTTP_200_OK)

        return Response({'error': 'User not recognized'}, status=status.HTTP_404_NOT_FOUND)


# Page render (no CSRF issue for GET)
def attendance_page(request):
    return render(request, 'attendance.html')