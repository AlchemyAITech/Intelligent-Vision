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
        
        # Internal Cache for recognized faces
        self.known_embeddings = []
        self.known_names = []
        self.load_face_bank()

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
        """Loads face embeddings from the face_bank directory."""
        self.known_embeddings = []
        self.known_names = []
        if not os.path.exists(self.face_bank_path):
            return

        for person_name in os.listdir(self.face_bank_path):
            person_dir = os.path.join(self.face_bank_path, person_name)
            if not os.path.isdir(person_dir) or person_name == "Strangers":
                continue
            
            for img_name in os.listdir(person_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(person_dir, img_name)
                    embedding = self.get_embedding_from_file(img_path)
                    if embedding is not None:
                        self.known_embeddings.append(embedding)
                        self.known_names.append(person_name)
        
        strangers_dir = os.path.join(self.face_bank_path, "Strangers")
        if os.path.exists(strangers_dir):
            for person_name in os.listdir(strangers_dir):
                person_dir = os.path.join(strangers_dir, person_name)
                if not os.path.isdir(person_dir): continue
                for img_name in os.listdir(person_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        embedding = self.get_embedding_from_file(os.path.join(person_dir, img_name))
                        if embedding is not None:
                            self.known_embeddings.append(embedding)
                            self.known_names.append(person_name)
                            
        print(f"Loaded {len(self.known_names)} faces from bank.")

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

    def get_hand_analysis(self, image, do_landmarks=True, do_gesture=True):
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
                custom_name, dist = self.match_custom_gesture(hand['landmarks'])
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
        count = len(os.listdir(person_dir))
        img_path = os.path.join(person_dir, f"face_{count}.jpg")
        cv2.imwrite(img_path, face_crop)
        self.load_face_bank() # Refresh bank
        return img_path

    def draw_results(self, image, faces=None, face_landmarks=None, recognized_names=None, hands=None):
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
                    cv2.circle(res_img, (int(pt[0]), int(pt[1])), 3, (0, 255, 255), -1)
        
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

