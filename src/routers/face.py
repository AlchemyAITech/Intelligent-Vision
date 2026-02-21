import cv2
import numpy as np
import os
import torch
import json
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.distance import cosine
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import hashlib

class FaceManager:
    def __init__(self, face_bank_path="data/face_bank", models_path="models"):
        self.face_bank_path = face_bank_path
        self.models_path = models_path
        os.makedirs(self.face_bank_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)
        
        # Initialize Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"FaceManager using device: {self.device}")
        
        # MediaPipe Tasks Initialization
        self.init_mediapipe_tasks()
        
        # Initialize MTCNN (Backup for legacy or specialized crops)
        self.mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=self.device
        )
        
        # Initialize InceptionResnetV1 for Recognition (Embeddings)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # Blacklist Management
        self.blacklist_path = os.path.join(self.face_bank_path, "blacklist.json")
        self.blacklist = self.load_blacklist()
        
        # Gesture Mapping (Standard Gestures)
        self.gesture_map_path = os.path.join(self.face_bank_path, "gesture_map.json")
        self.gesture_map = self.load_gesture_map()

        # Custom Gesture Recording (Template Matching)
        self.custom_gestures_path = os.path.join(self.face_bank_path, "custom_gestures.json")
        self.custom_gestures = self.load_custom_gestures()
        
        # Embedding Cache
        self.cache_path = os.path.join(self.face_bank_path, "embeddings_cache.json")
        self.cache = self.load_cache()
        
        # Internal Cache for recognized faces
        self.known_embeddings = []
        self.known_names = []
        self.load_face_bank()

    def load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
                    else:
                        print("FaceManager: Cache format invalid, resetting.")
                        return {}
            except (json.JSONDecodeError, Exception) as e:
                print(f"FaceManager: Error loading cache ({e}), resetting.")
                return {}
        return {}

    def save_cache(self):
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def init_mediapipe_tasks(self):
        """Initializes the MediaPipe Tasks API for Face and Hands."""
        try:
            # 1. Face Detector
            detector_model_path = os.path.join(self.models_path, "face_detector.tflite")
            if os.path.exists(detector_model_path):
                base_options_det = python.BaseOptions(model_asset_path=detector_model_path)
                options_det = vision.FaceDetectorOptions(base_options=base_options_det)
                self.detector = vision.FaceDetector.create_from_options(options_det)
            else:
                self.detector = None

            # 2. Face Landmarker (478 points)
            landmarker_model_path = os.path.join(self.models_path, "face_landmarker.task")
            if os.path.exists(landmarker_model_path):
                base_options_land = python.BaseOptions(model_asset_path=landmarker_model_path)
                options_land = vision.FaceLandmarkerOptions(
                    base_options=base_options_land,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    num_faces=5
                )
                self.landmarker = vision.FaceLandmarker.create_from_options(options_land)
            else:
                self.landmarker = None

            # 3. Hand Landmarker
            hand_model_path = os.path.join(self.models_path, "hand_landmarker.task")
            if os.path.exists(hand_model_path):
                base_options_hand = python.BaseOptions(model_asset_path=hand_model_path)
                options_hand = vision.HandLandmarkerOptions(base_options=base_options_hand, num_hands=2)
                self.hand_landmarker = vision.HandLandmarker.create_from_options(options_hand)
            else:
                self.hand_landmarker = None

            # 4. Gesture Recognizer
            gesture_model_path = os.path.join(self.models_path, "gesture_recognizer.task")
            if os.path.exists(gesture_model_path):
                base_options_gest = python.BaseOptions(model_asset_path=gesture_model_path)
                options_gest = vision.GestureRecognizerOptions(base_options=base_options_gest)
                self.gesture_recognizer = vision.GestureRecognizer.create_from_options(options_gest)
            else:
                self.gesture_recognizer = None

            print("MediaPipe Tasks (Face, Hands, Gestures) initialized.")
        except Exception as e:
            print(f"Error initializing MediaPipe Tasks: {e}")
            self.detector = None
            self.landmarker = None
            self.hand_landmarker = None
            self.gesture_recognizer = None

    def load_blacklist(self):
        """Loads the blacklist from a JSON file."""
        if os.path.exists(self.blacklist_path):
            with open(self.blacklist_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def save_blacklist(self):
        """Saves current blacklist to file."""
        with open(self.blacklist_path, 'w', encoding='utf-8') as f:
            json.dump(self.blacklist, f, ensure_ascii=False, indent=2)

    def load_gesture_map(self):
        """Loads the gesture-to-meaning map."""
        default_map = {
            "Thumb_Up": "点赞 (Good)",
            "Thumb_Down": "反对 (Bad)",
            "Victory": "胜利 (Peace)",
            "OK": "确认 (OK)",
            "Closed_Fist": "拳头 (Fist)",
            "Open_Palm": "手掌 (Palm)",
            "Pointing_Up": "指向上方",
            "ILoveYou": "🤟 爱你 (Love)",
            "None": "无手势"
        }
        if os.path.exists(self.gesture_map_path):
            try:
                with open(self.gesture_map_path, 'r', encoding='utf-8') as f:
                    user_map = json.load(f)
                    # Merge with default to ensure all keys exist
                    default_map.update(user_map)
            except: pass
        return default_map

    def load_custom_gestures(self):
        """Loads custom gesture templates."""
        if os.path.exists(self.custom_gestures_path):
            with open(self.custom_gestures_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_gesture_map(self, new_map):
        """Saves gesture mapping to JSON."""
        self.gesture_map = new_map
        with open(self.gesture_map_path, 'w', encoding='utf-8') as f:
            json.dump(self.gesture_map, f, ensure_ascii=False, indent=4)

    def save_custom_gestures(self):
        """Saves custom gestures to JSON."""
        with open(self.custom_gestures_path, 'w', encoding='utf-8') as f:
            json.dump(self.custom_gestures, f, ensure_ascii=False, indent=4)

    def save_custom_gesture(self, landmarks, name):
        """Saves a single custom gesture template."""
        # Normalize
        normalized = self.normalize_landmarks(landmarks)
        # Store as list
        self.custom_gestures[name] = normalized.tolist()
        self.save_custom_gestures()


    @staticmethod
    def normalize_landmarks(landmarks):
        """Normalizes landmarks: Translation -> Scaling. Handles list of dicts/objects."""
        # Convert to numpy array safely
        pts_list = []
        for lm in landmarks:
            if hasattr(lm, 'x'):
                pts_list.append([lm.x, lm.y, lm.z])
            elif isinstance(lm, dict) and 'x' in lm:
                 pts_list.append([lm['x'], lm['y'], lm['z']])
            elif isinstance(lm, (list, tuple)) and len(lm) >= 2:
                 # Ensure 3D
                 if len(lm) == 2:
                     pts_list.append([lm[0], lm[1], 0.0])
                 else:
                     pts_list.append([lm[0], lm[1], lm[2]])
        
        pts = np.array(pts_list, dtype=np.float32)
        
        if len(pts) == 0:
            return pts

        # 1. Translate (Wrist at 0,0)
        wrist = pts[0]
        pts = pts - wrist
        # 2. Scale (Max dist to 1.0)
        max_dist = np.max(np.linalg.norm(pts, axis=1))
        if max_dist > 0:
            pts = pts / max_dist
        return pts

    def match_custom_gesture(self, landmarks, threshold=0.15):
        """Matches a hand against custom templates using Euclidean distance."""
        if not self.custom_gestures:
            return None, 1.0
            
        current = self.normalize_landmarks(landmarks)
        best_name = None
        min_dist = 100.0
        
        for name, template_list in self.custom_gestures.items():
            template = np.array(template_list)
            # Simple Euclidean distance sum
            dist = np.mean(np.linalg.norm(current - template, axis=1))
            if dist < min_dist:
                min_dist = dist
                best_name = name
                
        if min_dist < threshold:
            return best_name, min_dist
        return None, min_dist

    def load_face_bank(self):
        """Loads face embeddings from the face_bank directory with internal caching."""
        self.known_embeddings = []
        self.known_names = []
        if not os.path.exists(self.face_bank_path):
            return

        print(f"FaceManager: Scanning face bank at {self.face_bank_path}...")
        
        updated = False
        # 1. Normal People
        for person_name in os.listdir(self.face_bank_path):
            person_dir = os.path.join(self.face_bank_path, person_name)
            if not os.path.isdir(person_dir) or person_name == "Strangers":
                continue
            
            for img_name in os.listdir(person_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(person_dir, img_name)
                    embedding, was_cached = self.get_embedding_cached(img_path)
                    if not was_cached: updated = True
                    if embedding is not None:
                        self.known_embeddings.append(embedding)
                        self.known_names.append(person_name)
        
        # 2. Strangers
        strangers_dir = os.path.join(self.face_bank_path, "Strangers")
        if os.path.exists(strangers_dir):
            for person_name in os.listdir(strangers_dir):
                person_dir = os.path.join(strangers_dir, person_name)
                if not os.path.isdir(person_dir): continue
                for img_name in os.listdir(person_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(person_dir, img_name)
                        embedding, was_cached = self.get_embedding_cached(img_path)
                        if not was_cached: updated = True
                        if embedding is not None:
                            self.known_embeddings.append(embedding)
                            self.known_names.append(person_name)
        
        if updated:
            self.save_cache()
            
        # 3. Clean up obsolete cache entries (files that no longer exist)
        obsolete_keys = []
        for key in self.cache.keys():
            # key format: path_mtime
            # We need to extract the path. Split from the last underscore
            if "_" in key:
                img_path = key.rsplit("_", 1)[0]
                if not os.path.exists(img_path):
                    obsolete_keys.append(key)
        
        if obsolete_keys:
            print(f"FaceManager: Pruning {len(obsolete_keys)} obsolete cache entries.")
            for k in obsolete_keys:
                del self.cache[k]
            self.save_cache()

        print(f"FaceManager: Loaded {len(self.known_names)} faces.")

    def get_embedding_cached(self, img_path):
        """Returns embedding from cache if valid, else computes and caches."""
        mtime = os.path.getmtime(img_path)
        # Unique key based on path and size/mtime
        cache_key = f"{img_path}_{mtime}"
        
        if cache_key in self.cache:
            return np.array(self.cache[cache_key]), True
            
        embedding = self.get_embedding_from_file(img_path)
        if embedding is not None:
            self.cache[cache_key] = embedding.tolist()
            return embedding, False
        return None, False

    def get_embedding_from_file(self, img_path):
        """Extracts embedding from an image file."""
        try:
            img = Image.open(img_path).convert('RGB')
            face = self.mtcnn(img)
            if face is not None:
                with torch.no_grad():
                    embedding = self.resnet(face.unsqueeze(0).to(self.device)).cpu().numpy()[0]
                return embedding
            else:
                img = img.resize((160, 160))
                img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float().to(self.device) / 255.0
                img_tensor = (img_tensor - 0.5) / 0.5
                with torch.no_grad():
                    embedding = self.resnet(img_tensor.unsqueeze(0)).cpu().numpy()[0]
                return embedding
        except Exception as e:
            print(f"Error loading embedding for {img_path}: {e}")
            return None

    def detect_faces(self, image):
        """Detects faces."""
        if self.detector:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            detection_result = self.detector.detect(mp_image)
            faces = []
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                faces.append({
                    'bbox': [int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)],
                    'score': float(detection.categories[0].score)
                })
            return faces
        else:
            img_rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            boxes, probs, points = self.mtcnn.detect(img_rgb, landmarks=True)
            faces = []
            if boxes is not None:
                for i, box in enumerate(boxes):
                    x, y, x2, y2 = box
                    faces.append({
                        'bbox': [int(x), int(y), int(x2-x), int(y2-y)],
                        'score': float(probs[i]),
                        'basic_landmarks': points[i].tolist() if points is not None else None
                    })
            return faces

    def get_face_landmarks(self, image):
        """Extracts dense landmarks (478 pts)."""
        if self.landmarker:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            result = self.landmarker.detect(mp_image)
            h, w = image.shape[:2]
            all_landmarks = []
            for face_landmarks in result.face_landmarks:
                pts = [[int(l.x * w), int(l.y * h)] for l in face_landmarks]
                all_landmarks.append(pts)
            return all_landmarks
        return None

    def get_hand_analysis(self, image, do_landmarks=True, do_gesture=True, gesture_threshold=0.15):
        """Extracts hand landmarks and recognized gestures."""
        h, w = image.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        hands_data = []
        
        # 1. Landmarks
        if do_landmarks and self.hand_landmarker:
            result = self.hand_landmarker.detect(mp_image)
            for i, hand_landmarks in enumerate(result.hand_landmarks):
                pts = [[int(l.x * w), int(l.y * h)] for l in hand_landmarks]
                hands_data.append({'landmarks': pts, 'gesture': None})
        
        # 2. Gestures
        if do_gesture and self.gesture_recognizer:
            result = self.gesture_recognizer.recognize(mp_image)
            # If we already have landmarks from hand_landmarker, we might want to align.
            # However, gesture_recognizer returns its own results.
            # We'll merge or just use gesture result directly.
            for i, gesture in enumerate(result.gestures):
                cat = gesture[0].category_name
                target_hand_idx = i if i < len(hands_data) else -1
                meaning = self.gesture_map.get(cat, cat)
                
                if target_hand_idx != -1:
                    hands_data[target_hand_idx]['gesture'] = meaning
                else:
                    hands_data.append({'landmarks': None, 'gesture': meaning})
        
        # 3. Custom Gesture Matching (Few-shot)
        for hand in hands_data:
            if hand['landmarks'] and (not hand['gesture'] or hand['gesture'] == "无手势"):
                custom_name, dist = self.match_custom_gesture(hand['landmarks'], threshold=gesture_threshold)
                if custom_name:
                    hand['gesture'] = f"{custom_name} (自定义)"
                    
        return hands_data

    def get_embedding(self, face_crop):
        """Generates embedding for a cropped face image."""
        try:
            face_img = cv2.resize(face_crop, (160, 160))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.tensor(face_img).permute(2, 0, 1).float().to(self.device) / 255.0
            img_tensor = (img_tensor - 0.5) / 0.5
            with torch.no_grad():
                embedding = self.resnet(img_tensor.unsqueeze(0)).cpu().numpy()[0]
            return embedding
        except Exception:
            return None

    def recognize_face(self, embedding, threshold=0.6):
        """Matches embedding against the known faces."""
        if not self.known_embeddings:
            return "Unknown", 1.0
        
        min_dist = 1.0
        best_match = "Unknown"
        
        for i, known_emb in enumerate(self.known_embeddings):
            dist = cosine(embedding, known_emb)
            if dist < min_dist:
                min_dist = dist
                best_match = self.known_names[i]
        
        if min_dist < threshold:
            return best_match, min_dist
        else:
            return "Unknown", min_dist

    def save_new_face(self, face_crop, name):
        """Saves a new face to the bank. Unknows go into Strangers."""
        if name.startswith("Stranger_"):
            person_dir = os.path.join(self.face_bank_path, "Strangers", name)
        else:
            person_dir = os.path.join(self.face_bank_path, name)
            
        os.makedirs(person_dir, exist_ok=True)
        
        # Calculate SHA256 of image content for name collision prevention and deduplication
        ret, buffer = cv2.imencode('.jpg', face_crop)
        img_bytes = buffer.tobytes()
        img_hash = hashlib.sha256(img_bytes).hexdigest()[:16] # Use first 16 chars
        
        img_path = os.path.join(person_dir, f"{img_hash}.jpg")
        
        # If exists, we skip writing (De-duplication)
        if not os.path.exists(img_path):
            cv2.imwrite(img_path, face_crop)
            
        self.load_face_bank() # Refresh bank
        return img_path

    def draw_results(self, image, faces=None, face_landmarks=None, recognized_names=None, hands=None, target_name=None):
        """Draws results. Red for blacklist. Support Chinese & Hands."""
        from PIL import Image, ImageDraw, ImageFont
        
        is_threat = False
        h_img, w_img = image.shape[:2]
        
        # Load Chinese font
        font_path = "/System/Library/Fonts/STHeiti Medium.ttc"
        if not os.path.exists(font_path):
            font_path = "/Library/Fonts/Arial Unicode.ttf" # Fallback
            
        # Convert BGR to RGB for PIL
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # Determine Font Size
        try:
            font_size = max(20, int(h_img / 25)) 
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()
            
        # 1. Draw Faces
        if faces:
            for i, face in enumerate(faces):
                x, y, w, h = face['bbox']
                color = (0, 255, 0) # Green (RGB for PIL)
                
                # Determine Name & Color
                label = "未知" 
                if recognized_names and i < len(recognized_names):
                    name, dist = recognized_names[i]
                    if name == "Unknown":
                        label = "未知"
                        color = (255, 255, 0) # 黄色
                    elif name in self.blacklist:
                        color = (255, 0, 0) # 红色
                        is_threat = True
                        label = f"危险警告: {name}"
                    else:
                        label = f"{name} ({dist:.2f})"
                
                # SPECIAL HANDLING: If target_name is provided and result is Unknown, show "Enrolling..."
                if target_name and label == "未知":
                    label = f"正在录入: {target_name}"
                    color = (0, 122, 255) # Blue for enrolling state
                
                # Draw rect with PIL
                draw.rectangle([x, y, x + w, y + h], outline=color, width=4)
                
                # Draw Label Background & Text
                tw, th = draw.textbbox((0, 0), label, font=font)[2:]
                draw.rectangle([x, y - th - 10, x + tw + 10, y], fill=color)
                draw.text((x + 5, y - th - 5), label, font=font, fill=(0, 0, 0))

        # 2. Draw Hands & Gestures
        if hands:
            # MediaPipe Hand Connections
            HAND_CONNECTIONS = [
                (0, 1), (1, 2), (2, 3), (3, 4), # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8), # Index
                (5, 9), (9, 10), (10, 11), (11, 12), # Middle
                (9, 13), (13, 14), (14, 15), (15, 16), # Ring
                (13, 17), (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
                (5, 9), (9, 13), (13, 17) # Palm
            ]
            for hand in hands:
                hand_color = (0, 255, 255) # Yellow/Cyan
                if hand['landmarks']:
                    pts = hand['landmarks']
                    # Draw Connections (Skeletal Lines)
                    for connection in HAND_CONNECTIONS:
                        start_idx, end_idx = connection
                        if start_idx < len(pts) and end_idx < len(pts):
                            draw.line([pts[start_idx][0], pts[start_idx][1], pts[end_idx][0], pts[end_idx][1]], fill=hand_color, width=3)
                            
                    # Draw Points
                    for pt in pts:
                        draw.ellipse([pt[0]-4, pt[1]-4, pt[0]+4, pt[1]+4], fill=hand_color)
                
                if hand['gesture']:
                    # Draw gesture text near the first landmark or center
                    g_text = hand['gesture']
                    if hand['landmarks']:
                        pos = (hand['landmarks'][0][0], hand['landmarks'][0][1] - 40)
                    else:
                        pos = (10, 50) # Fallback
                    
                    tw, th = draw.textbbox((0, 0), g_text, font=font)[2:]
                    draw.rectangle([pos[0]-5, pos[1]-5, pos[0]+tw+5, pos[1]+th+5], fill=(0, 0, 0, 180))
                    draw.text(pos, g_text, font=font, fill=hand_color)

        # Convert back to BGR for OpenCV landmark drawing (if any)
        res_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # 3. Draw Face Landmarks (OpenCV)
        if face_landmarks: # Dense or Basic
            for pts in face_landmarks:
                for pt in pts:
                    cv2.circle(res_img, (int(pt[0]), int(pt[1])), 1, (0, 255, 255), -1)
        
        return res_img, is_threat

    def apply_red_glow(self, image):
        """Draws a red breathing frame around the whole image to indicate threat."""
        import time
        import math
        
        # Calculate pulse intensity using sine wave (0.0 to 1.0)
        # Period = 1.0s (frequency = 1Hz)
        t = time.time()
        intensity = (math.sin(t * 2 * math.pi) + 1) / 2
        
        overlay = image.copy()
        lw = int(30 + 30 * intensity) # Thickness also pulses
        cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), (0, 0, 255), lw)
        
        # Alpha blending for soft breathing effect
        alpha = 0.3 + 0.5 * intensity
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        return image

    def delete_face_sample(self, file_path):
        """Deletes a specific face sample and reloads the bank."""
        if os.path.exists(file_path):
            os.remove(file_path)
            self.load_face_bank()
            return True
        return False



import os
import shutil
import io
import base64
import numpy as np
import cv2
from PIL import Image
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import time

router = APIRouter()
face_manager = FaceManager()

# --- Management Endpoints ---

@router.post("/bank/enroll")
async def enroll_face(
    name: str = Form(...),
    image_b64: str = Form(...)
):
    """Enrolls a face crop from base64 string with auto-detection."""
    try:
        header, encoded = image_b64.split(",", 1) if "," in image_b64 else (None, image_b64)
        img_data = base64.b64decode(encoded)
        nparr = np.frombuffer(img_data, np.uint8)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_np is None:
             raise ValueError("Invalid image data.")
        
        # Enforce Face Detection & Cropping
        faces = face_manager.detect_faces(image_np)
        if not faces:
            raise ValueError("未检测到有效人脸！请提供特征清晰的人像！")
        
        # Take the largest face if multiple
        faces.sort(key=lambda f: f['bbox'][2] * f['bbox'][3], reverse=True)
        x, y, w, h = faces[0]['bbox']
        x, y = max(0, x), max(0, y)
        face_crop = image_np[y:y+h, x:x+w]
        
        if face_crop.size == 0:
             raise ValueError("Face crop failed.")
             
        path = face_manager.save_new_face(face_crop, name)
        return {"status": "ok", "path": path}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/bank/capture_stranger")
async def capture_stranger(
    file: UploadFile = File(...),
    bbox_json: str = Form(...)
):
    """Captures a face from a full image given a bounding box and saves as a new stranger."""
    try:
        import json
        bbox = json.loads(bbox_json)
        contents = await file.read()
        image_pil = Image.open(io.BytesIO(contents)).convert('RGB')
        image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        x, y, w, h = bbox
        x, y = max(0, x), max(0, y)
        face_crop = image_np[y:y+h, x:x+w]
        
        if face_crop.size == 0:
             raise ValueError("Empty face crop.")
             
        stranger_id = f"Stranger_{int(time.time())}"
        path = face_manager.save_new_face(face_crop, stranger_id)
        return {"status": "ok", "name": stranger_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/bank/list")
async def list_face_bank():
    """Lists all persons and their sample counts in the face bank."""
    bank_path = face_manager.face_bank_path
    if not os.path.exists(bank_path): return []
    
    results = []
    # Known people
    for person in sorted(os.listdir(bank_path)):
        p_path = os.path.join(bank_path, person)
        if os.path.isdir(p_path) and person != "Strangers":
            count = len([f for f in os.listdir(p_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            results.append({
                "name": person, 
                "count": count, 
                "is_blacklist": person in face_manager.blacklist,
                "is_stranger": False
            })
    
    # Strangers
    strangers_path = os.path.join(bank_path, "Strangers")
    if os.path.exists(strangers_path):
        for person in sorted(os.listdir(strangers_path)):
            p_path = os.path.join(strangers_path, person)
            if os.path.isdir(p_path):
                 count = len([f for f in os.listdir(p_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
                 results.append({
                     "name": person, 
                     "count": count, 
                     "is_blacklist": False,
                     "is_stranger": True
                 })
    return results

@router.post("/bank/toggle_blacklist")
async def toggle_blacklist(name: str = Form(...)):
    if name in face_manager.blacklist:
        face_manager.blacklist.remove(name)
    else:
        face_manager.blacklist.append(name)
    face_manager.save_blacklist()
    return {"status": "ok", "is_blacklist": name in face_manager.blacklist}

@router.post("/bank/promote_stranger")
async def promote_stranger(name: str = Form(...), new_name: str = Form(...)):
    """Moves a person from Strangers to the main face bank."""
    stranger_p = os.path.join(face_manager.face_bank_path, "Strangers", name)
    known_p = os.path.join(face_manager.face_bank_path, new_name)
    
    if os.path.exists(stranger_p):
        import shutil
        # Cache Sync Prep
        old_keys_to_update = {}
        for filename in os.listdir(stranger_p):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                old_img_path = os.path.join(stranger_p, filename)
                for key in list(face_manager.cache.keys()):
                    if key.startswith(old_img_path + "_"):
                        old_keys_to_update[key] = filename
                        break

        if os.path.exists(known_p):
            # Merge folders with de-duplication
            for f in os.listdir(stranger_p):
                s_file = os.path.join(stranger_p, f)
                t_file = os.path.join(known_p, f)
                if os.path.exists(t_file):
                    # Duplicate (same hash likely), remove stranger version
                    # Remove from cache if tracked
                    mtime = os.path.getmtime(s_file)
                    key = f"{s_file}_{mtime}"
                    if key in face_manager.cache: del face_manager.cache[key]
                    os.remove(s_file)
                else:
                    shutil.move(s_file, t_file)
            if not os.listdir(stranger_p):
                os.rmdir(stranger_p)
        else:
            os.rename(stranger_p, known_p)
        
        # Sync Cache
        if old_keys_to_update:
            for old_key, filename in old_keys_to_update.items():
                if old_key in face_manager.cache:
                    emb_data = face_manager.cache.pop(old_key)
                    new_img_path = os.path.join(known_p, filename)
                    if os.path.exists(new_img_path):
                        new_mtime = os.path.getmtime(new_img_path)
                        new_key = f"{new_img_path}_{new_mtime}"
                        face_manager.cache[new_key] = emb_data
            face_manager.save_cache()

        face_manager.load_face_bank()
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Stranger not found")

@router.get("/bank/gestures")
async def list_gestures():
    # Provide predefined gesture defaults (empty mean no alias set)
    defaults = {
        "None": "",
        "Closed_Fist": "",
        "Open_Palm": "",
        "Pointing_Up": "",
        "Thumb_Down": "",
        "Thumb_Up": "",
        "Victory": "",
        "ILoveYou": ""
    }
    aliases = {**defaults, **face_manager.gesture_map}
    return {
        "custom": face_manager.custom_gestures,
        "aliases": aliases
    }

@router.post("/bank/save_gesture_alias")
async def save_gesture_alias(name: str = Form(...), alias: str = Form(...)):
    face_manager.gesture_map[name] = alias
    face_manager.save_gesture_map(face_manager.gesture_map)
    return {"status": "ok"}

@router.post("/bank/save_gesture")
async def save_gesture(
    name: str = Form(...),
    landmarks_json: str = Form(...)
):
    try:
        import json
        landmarks = json.loads(landmarks_json)
        face_manager.save_custom_gesture(landmarks, name)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/bank/delete_gesture")
async def delete_gesture(name: str = Form(...)):
    if name in face_manager.custom_gestures:
        del face_manager.custom_gestures[name]
        face_manager.save_custom_gestures()
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Gesture not found")

@router.get("/bank/samples/{name:path}")
async def get_samples(name: str):
    """Lists all samples for a specific person."""
    p_path = os.path.join(face_manager.face_bank_path, name)
    if not os.path.exists(p_path): return []
    
    files = sorted([f for f in os.listdir(p_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    # Return absolute or relative paths that can be used to fetch the image
    # Note: main.py mounts /images to project root /images, but face_bank might be in data/face_bank
    # Let's check face_manager.face_bank_path
    return files

@router.post("/bank/rename")
async def rename_person(old_name: str = Form(...), new_name: str = Form(...)):
    bank_path = face_manager.face_bank_path
    
    # Check if this person is a stranger
    is_stranger = False
    old_p = os.path.join(bank_path, old_name)
    if not os.path.exists(old_p):
        old_p = os.path.join(bank_path, "Strangers", old_name)
        is_stranger = True

    if os.path.exists(old_p):
        new_p = os.path.join(bank_path, "Strangers", new_name) if is_stranger else os.path.join(bank_path, new_name)
        
        if old_p == new_p: return {"status": "ok"}
        
        import shutil
        # Cache Sync Prep
        old_keys_to_update = {}
        for filename in os.listdir(old_p):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                old_img_path = os.path.join(old_p, filename)
                for key in list(face_manager.cache.keys()):
                    if key.startswith(old_img_path + "_"):
                        old_keys_to_update[key] = filename
                        break
        
        if os.path.exists(new_p):
            # Merge logic for rename
            for f in os.listdir(old_p):
                s_file = os.path.join(old_p, f)
                t_file = os.path.join(new_p, f)
                if os.path.exists(t_file):
                    os.remove(s_file) # Duplicate
                else:
                    shutil.move(s_file, t_file)
            if not os.listdir(old_p):
                os.rmdir(old_p)
        else:
            os.rename(old_p, new_p)
        
        # Apply cache updates
        if old_keys_to_update:
            for old_key, filename in old_keys_to_update.items():
                if old_key in face_manager.cache:
                    emb_data = face_manager.cache.pop(old_key)
                    new_img_path = os.path.join(new_p, filename)
                    if os.path.exists(new_img_path):
                        new_mtime = os.path.getmtime(new_img_path)
                        new_key = f"{new_img_path}_{new_mtime}"
                        face_manager.cache[new_key] = emb_data
            face_manager.save_cache()
        
        # Update blacklist reference if needed
        if old_name in face_manager.blacklist:
            face_manager.blacklist.remove(old_name)
            face_manager.blacklist.append(new_name)
            face_manager.save_blacklist()
            
        face_manager.load_face_bank()
        return {"status": "ok"}
        
    raise HTTPException(status_code=404, detail="Person not found")

@router.post("/bank/delete_sample")
async def delete_sample(name: str = Form(...), filename: str = Form(...)):
    target = os.path.join(face_manager.face_bank_path, name, filename)
    if os.path.exists(target):
        # Cache Sync: Remove key
        mtime = os.path.getmtime(target)
        key = f"{target}_{mtime}"
        if key in face_manager.cache:
            del face_manager.cache[key]
            face_manager.save_cache()
            
        os.remove(target)
        face_manager.load_face_bank()
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="File not found")

@router.post("/bank/transfer_sample")
async def transfer_sample(
    name: str = Form(...), 
    filename: str = Form(...), 
    new_name: str = Form(...)
):
    source = os.path.join(face_manager.face_bank_path, name, filename)
    if not os.path.exists(source):
        raise HTTPException(status_code=404, detail="Source file not found")
        
    target_dir = os.path.join(face_manager.face_bank_path, new_name)
    os.makedirs(target_dir, exist_ok=True)
    
    # Standardize filename to Hash to prune legacy face_N names during transfer
    with open(source, 'rb') as f:
        img_bytes = f.read()
    img_hash = hashlib.sha256(img_bytes).hexdigest()[:16]
    unique_filename = f"{img_hash}.jpg"
    
    target = os.path.join(target_dir, unique_filename)
    
    if source == target:
        return {"status": "ok", "new_filename": unique_filename}
        
    # Handle Collision/De-duplication
    if os.path.exists(target):
        # Same image already exist in target, just remove source (De-duplication)
        # Sync Cache first: Ensure any recognition pointing to source is cleared
        old_mtime = os.path.getmtime(source)
        old_key = f"{source}_{old_mtime}"
        if old_key in face_manager.cache:
            del face_manager.cache[old_key]
        
        os.remove(source)
    else:
        import shutil
        # Capture old cache info to sync
        old_mtime = os.path.getmtime(source)
        old_key = f"{source}_{old_mtime}"
        
        shutil.move(source, target)
        
        # Sync Cache: Update the key to the new location
        new_mtime = os.path.getmtime(target)
        new_key = f"{target}_{new_mtime}"
        
        if old_key in face_manager.cache:
            emb_data = face_manager.cache.pop(old_key)
            face_manager.cache[new_key] = emb_data
            face_manager.save_cache()
    
    face_manager.load_face_bank()
    return {"status": "ok", "new_filename": unique_filename}

@router.post("/bank/delete_person")
async def delete_person(name: str = Form(...)):
    bank_path = face_manager.face_bank_path
    target_dir = os.path.join(bank_path, name)
    # Also support Strangers dir deletion
    if not os.path.exists(target_dir):
        target_dir = os.path.join(bank_path, "Strangers", name)
        
    if os.path.exists(target_dir) and os.path.isdir(target_dir):
        # Cache Sync: Remove all keys starting with this directory path
        keys_to_del = [k for k in face_manager.cache.keys() if k.startswith(target_dir + os.sep)]
        for k in keys_to_del:
            del face_manager.cache[k]
        if keys_to_del:
            face_manager.save_cache()
            
        import shutil
        shutil.rmtree(target_dir)
        # Check if in blacklist, remove
        if name in face_manager.blacklist:
            face_manager.blacklist.remove(name)
            face_manager.save_blacklist()
        face_manager.load_face_bank()
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Person directory not found")

@router.post("/bank/upload_sample")
async def upload_sample(
    name: str = Form(...), 
    file: UploadFile = File(...)
):
    try:
        contents = await file.read()
        image_pil = Image.open(io.BytesIO(contents)).convert('RGB')
        image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        # Enforce Face Detection & Cropping to keep bank clean and models accurate
        faces = face_manager.detect_faces(image_np)
        if not faces:
            raise ValueError("No face detected in the provided image. Please ensure the face is clearly visible.")
        
        # Take the largest face if multiple
        faces.sort(key=lambda f: f['bbox'][2] * f['bbox'][3], reverse=True)
        x, y, w, h = faces[0]['bbox']
        x, y = max(0, x), max(0, y)
        face_crop = image_np[y:y+h, x:x+w]
        
        if face_crop.size == 0:
            raise ValueError("Face crop failed due to boundary issues.")

        face_manager.save_new_face(face_crop, name)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- Analysis Endpoints ---

def encode_pil_base64(pil_img: Image.Image, format="JPEG") -> str:
    buffered = io.BytesIO()
    pil_img.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

@router.post("/analyze_metadata")
async def face_analyze_metadata(
    file: UploadFile = File(...),
    do_hands: bool = Form(True),
    gest_threshold: float = Form(0.15)
):
    """Specific endpoint for metadata only extraction (no image plotting)."""
    try:
        import asyncio
        contents = await file.read()
        loop = asyncio.get_event_loop()

        def run_sync_meta():
            image_pil = Image.open(io.BytesIO(contents)).convert('RGB')
            image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            
            result = {}
            if do_hands:
                hands = face_manager.get_hand_analysis(image_np, gesture_threshold=gest_threshold)
                if hands:
                    result["hand_landmarks"] = hands[0]['landmarks']
            return result

        return await loop.run_in_executor(None, run_sync_meta)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/analyze")
async def face_analyze(
    file: UploadFile = File(...),
    do_recognition: bool = Form(True),
    do_landmarks: bool = Form(True),
    do_hands: bool = Form(True),
    rec_threshold: float = Form(0.6),
    gest_threshold: float = Form(0.15),
    target_name: str = Form(None)
):
    try:
        import asyncio
        contents = await file.read()
        loop = asyncio.get_event_loop()

        def run_sync_analyze():
            image_pil = Image.open(io.BytesIO(contents)).convert('RGB')
            image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            
            # 1. Detect faces (Only if recognition or landmarks are requested)
            faces = []
            if do_recognition or do_landmarks:
                faces = face_manager.detect_faces(image_np)
            
            # 2. Recognize strings
            recognized_names = []
            if do_recognition:
                for face in faces:
                    x, y, w, h = face['bbox']
                    x, y = max(0, x), max(0, y)
                    face_crop = image_np[y:y+h, x:x+w]
                    if face_crop.size > 0:
                        emb = face_manager.get_embedding(face_crop)
                        if emb is not None:
                            name, dist = face_manager.recognize_face(emb, threshold=rec_threshold)
                            recognized_names.append((name, dist))
                        else:
                            recognized_names.append(("Unknown", 1.0))
                    else:
                        recognized_names.append(("Unknown", 1.0))
            
            # 3. Landmarks
            landmarks = None
            if do_landmarks:
                landmarks = face_manager.get_face_landmarks(image_np)
                
            # 4. Hands
            hands = None
            if do_hands:
                hands = face_manager.get_hand_analysis(image_np, gesture_threshold=gest_threshold)
                
            # Draw
            plotted_np, is_threat = face_manager.draw_results(
                image_np, 
                faces=faces, 
                face_landmarks=landmarks, 
                recognized_names=recognized_names,
                hands=hands,
                target_name=target_name
            )
            
            if is_threat:
                plotted_np = face_manager.apply_red_glow(plotted_np)
                
            return {
                "image_b64": encode_pil_base64(Image.fromarray(cv2.cvtColor(plotted_np, cv2.COLOR_BGR2RGB))),
                "faces_count": len(faces),
                "is_threat": is_threat,
                "faces": faces, 
                "hands": hands
            }

        return await loop.run_in_executor(None, run_sync_analyze)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
