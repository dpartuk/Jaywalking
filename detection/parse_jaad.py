import os
import cv2
import xml.etree.ElementTree as ET
import random
from tqdm import tqdm

# --- CONFIGURATION ---
JAAD_ROOT = "./JAAD"
VIDEO_DIR = os.path.join(JAAD_ROOT, "JAAD_clips")
ANNOTATION_DIR = os.path.join(JAAD_ROOT, "annotations")
OUTPUT_DIR = "./dataset_yolo"
TRAIN_RATIO = 0.8

def setup_directories():
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

def parse_xml_to_dict(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    frame_data = {}

    for track in root.findall('track'):
        label = track.attrib['label']
        if label != 'pedestrian': 
            continue
            
        for box in track.findall('box'):
            frame_id = int(box.attrib['frame'])
            
            # --- CORRECTED LOGIC ---
            class_id = 1 # Default: Static (1)
            
            for attr in box.findall('attribute'):
                if attr.attrib['name'] == 'cross':
                    val = attr.text.lower().strip()
                    
                    if val == 'crossing':
                        class_id = 0 # Target: Crossing
                    # 'not-crossing' remains class_id = 1
                    break
            
            # Extract BBox
            xtl, ytl = float(box.attrib['xtl']), float(box.attrib['ytl'])
            xbr, ybr = float(box.attrib['xbr']), float(box.attrib['ybr'])
            
            if frame_id not in frame_data:
                frame_data[frame_id] = []
            
            frame_data[frame_id].append({
                'bbox': [xtl, ytl, xbr, ybr],
                'class_id': class_id
            })
            
    return frame_data

def process_video(video_filename, split):
    video_name = os.path.splitext(video_filename)[0]
    xml_path = os.path.join(ANNOTATION_DIR, f"{video_name}.xml")
    video_path = os.path.join(VIDEO_DIR, video_filename)
    
    if not os.path.exists(xml_path): return
        
    frames_with_peds = parse_xml_to_dict(xml_path)
    if not frames_with_peds: return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return 

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    current_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
            
        if current_frame in frames_with_peds:
            # Save Image
            img_filename = f"{video_name}_{current_frame}.jpg"
            save_path = os.path.join(OUTPUT_DIR, 'images', split, img_filename)
            cv2.imwrite(save_path, frame)
            
            # Save Label
            label_filename = f"{video_name}_{current_frame}.txt"
            label_path = os.path.join(OUTPUT_DIR, 'labels', split, label_filename)
            
            with open(label_path, 'w') as f:
                for ped in frames_with_peds[current_frame]:
                    cls = ped['class_id']
                    x1, y1, x2, y2 = ped['bbox']
                    
                    # Normalize
                    w = x2 - x1
                    h = y2 - y1
                    x_center = (x1 + (w / 2.0)) / width
                    y_center = (y1 + (h / 2.0)) / height
                    w /= width
                    h /= height
                    
                    f.write(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
        
        current_frame += 1
    cap.release()

def main():
    setup_directories()
    videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
    
    if not videos:
        print(f"❌ No videos found in {VIDEO_DIR}!")
        return

    random.shuffle(videos)
    split_idx = int(len(videos) * TRAIN_RATIO)
    train_videos = videos[:split_idx]
    val_videos = videos[split_idx:]
    
    print(f"Processing {len(train_videos)} Train videos...")
    for vid in tqdm(train_videos):
        process_video(vid, 'train')

    print(f"Processing {len(val_videos)} Val videos...")
    for vid in tqdm(val_videos):
        process_video(vid, 'val')
        
    print(f"\n✅ Done! Check {OUTPUT_DIR}")

if __name__ == "__main__":
    main()