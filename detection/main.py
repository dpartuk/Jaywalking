import cv2
import torch
import numpy as np
from collections import deque, Counter
from ultralytics import YOLO

# --- UNIVERSAL IMPORT ---
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
try:
    from transformers import AutoImageProcessor
except ImportError:
    from transformers import AutoFeatureExtractor as AutoImageProcessor

# --- CONFIGURATION ---
VIDEO_TO_TEST = "video_0025.mp4"  # <--- Changed to your specific video
VIDEO_PATH = f"./JAAD/JAAD_clips/{VIDEO_TO_TEST}"
OUTPUT_PATH = "final_output_texture.mp4"
YOLO_WEIGHTS = "runs/detect/jaad_yolo_run/weights/best.pt"

# --- TUNING KNOBS ---
# White Threshold: For standard stripes (0.08 = 8% coverage)
WHITE_THRESH = 0.08
# Edge Threshold: For triangles/bricks (0.05 = 5% pixels are edges)
EDGE_THRESH = 0.05 

# Class IDs
CLASS_ROAD = 6
CLASS_SIDEWALK = 11

# States
STATE_STATIC = "STATIC"
STATE_SAFE = "SAFE"
STATE_CROSSWALK = "CROSSWALK"
STATE_JAYWALKING = "JAYWALKING"

class PedestrianSystem:
    def __init__(self):
        print("🚀 Initializing System (Texture/Edge Mode)...")
        self.device = torch.device("mps")
        
        self.detector = YOLO(YOLO_WEIGHTS)
        
        model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
        try:
            self.seg_processor = AutoImageProcessor.from_pretrained(model_name)
        except:
            from transformers import AutoFeatureExtractor
            self.seg_processor = AutoFeatureExtractor.from_pretrained(model_name)

        self.seg_model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
        self.seg_model.to(self.device)
        self.seg_model.eval()
        
        self.track_history = {}
        print("✅ Models Loaded.")

    def get_segmentation_mask(self, frame):
        inputs = self.seg_processor(images=frame, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.seg_model(**inputs)
            logits = outputs.logits
            upsampled_logits = torch.nn.functional.interpolate(
                logits, size=frame.shape[:2], mode="bilinear", align_corners=False
            )
            mask = upsampled_logits.argmax(dim=1)[0]
        return mask

    def check_crossing_texture(self, frame, feet_x, feet_y):
        """
        New Hybrid Check:
        1. Checks for White Paint (Stripes)
        2. Checks for Edges/Texture (Triangles, Bricks, Pavers)
        """
        h, w, _ = frame.shape
        half_size = 20 # 40x40 patch
        x1, x2 = max(0, feet_x - half_size), min(w, feet_x + half_size)
        y1, y2 = max(0, feet_y - half_size), min(h, feet_y + half_size)
        
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0: return False, 0.0, 0.0

        # 1. Convert to Gray
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        
        # 2. Check White Density (Standard Zebra)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        white_ratio = cv2.countNonZero(binary) / binary.size
        
        # 3. Check Edge Density (Triangles/Bricks)
        # Canny finds the borders of shapes
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = cv2.countNonZero(edges) / edges.size
        
        # LOGIC: It's a crossing if it has Paint OR Texture
        is_crossing = (white_ratio > WHITE_THRESH) or (edge_ratio > EDGE_THRESH)
        
        return is_crossing, white_ratio, edge_ratio

    def get_smoothed_state(self, track_id, current_state):
        if track_id is None: return current_state
        if track_id not in self.track_history:
            self.track_history[track_id] = deque(maxlen=10)
        self.track_history[track_id].append(current_state)
        return Counter(self.track_history[track_id]).most_common(1)[0][0]

    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        print(f"🎬 Processing {input_path}...")
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            results = self.detector.track(frame, persist=True, verbose=False, device=self.device)
            boxes = results[0].boxes
            
            crossing_candidates = [b for b in boxes if int(b.cls[0]) == 0]
            seg_mask = self.get_segmentation_mask(frame) if crossing_candidates else None

            if boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
            else:
                track_ids = [None] * len(boxes)

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                
                raw_state = STATE_STATIC
                debug_white = 0.0
                debug_edge = 0.0
                surface_id = -1
                feet_x, feet_y = 0, 0

                if cls_id == 0: # Crossing
                    if seg_mask is not None:
                        feet_x = max(0, min(width-1, int((x1 + x2) / 2)))
                        feet_y = max(0, min(height-1, y2))
                        surface_id = seg_mask[feet_y, feet_x].item()
                        
                        # --- NEW HYBRID CHECK ---
                        is_valid_crossing, debug_white, debug_edge = self.check_crossing_texture(frame, feet_x, feet_y)
                        
                        if is_valid_crossing:
                            # 🟢 Priority 1: Texture/Paint found -> CROSSWALK
                            raw_state = STATE_CROSSWALK
                        elif surface_id == CLASS_SIDEWALK:
                            # 🟡 Priority 2: Sidewalk + No Texture -> SAFE
                            raw_state = STATE_SAFE
                        elif surface_id == CLASS_ROAD:
                            # 🔴 Priority 3: Road + No Texture -> JAYWALKING
                            raw_state = STATE_JAYWALKING
                        else:
                            raw_state = STATE_SAFE
                    else:
                        raw_state = STATE_JAYWALKING

                # Smooth
                final_state = self.get_smoothed_state(track_id, raw_state)

                # Colors
                if final_state == STATE_CROSSWALK:
                    color = (0, 255, 0) # Green
                    label = "CROSSWALK"
                elif final_state == STATE_JAYWALKING:
                    color = (0, 0, 255) # Red
                    label = "JAYWALKING"
                elif final_state == STATE_SAFE:
                    color = (255, 255, 0) # Cyan
                    label = "SAFE (SIDEWALK)"
                else: 
                    color = (128, 128, 128)
                    label = "STATIC"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                full_label = f"ID:{track_id} {label}"
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + 220, y1), color, -1)
                cv2.putText(frame, full_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                
                # --- NEW DEBUG OVERLAY ---
                if cls_id == 0 and seg_mask is not None:
                    cv2.rectangle(frame, (feet_x - 20, feet_y - 20), (feet_x + 20, feet_y + 20), (255, 0, 255), 1)
                    # Show BOTH metrics: W (White) and E (Edge)
                    # This lets you see which one triggers the Green Box
                    debug_text = f"W:{debug_white:.2f} | E:{debug_edge:.2f}"
                    cv2.putText(frame, debug_text, (feet_x + 25, feet_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            out.write(frame)
            frame_idx += 1
            if frame_idx % 30 == 0: print(f"Processed {frame_idx} frames...")

        cap.release()
        out.release()
        print(f"✅ Done! Video saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    system = PedestrianSystem()
    system.process_video(VIDEO_PATH, OUTPUT_PATH)