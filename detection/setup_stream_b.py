import torch
# We use the generic Auto classes which are much safer against version conflicts
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

def setup_context_expert():
    print("⬇️ Downloading SegFormer (ADE20K version)...")
    model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    
    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print("✅ Model & Processor Loaded Successfully.")
    
    # --- VERIFY CLASS IDs ---
    print("\n🔍 Verifying Class IDs for our Logic:")
    target_classes = ['crosswalk', 'sidewalk', 'road']
    found_ids = {}
    
    # Scan the labels
    for id, label in model.config.id2label.items():
        for target in target_classes:
            if target in label.lower():
                # We prioritize exact matches or the first valid hit
                if target not in found_ids:
                    found_ids[target] = (id, label)

    print("-" * 30)
    for target, (id, name) in found_ids.items():
        print(f"   • '{target}' is detected as Class ID {id} (Label: '{name}')")
    print("-" * 30)

if __name__ == "__main__":
    setup_context_expert()