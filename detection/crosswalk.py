from transformers import SegFormerImageProcessor, SegFormerForSemanticSegmentation

def check_crosswalk_id():
    # Load ADE20K pre-trained model
    model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    model = SegFormerForSemanticSegmentation.from_pretrained(model_name)
    
    # Print labels to find the exact ID
    print("Searching for crosswalk...")
    for id, label in model.config.id2label.items():
        if "cross" in label or "walk" in label:
            print(f"ID: {id} | Label: {label}")

# EXPECTED OUTPUT for ADE20K:
# ID: 12 | Label: sidewalk, pavement
# ID: 120 | Label: crosswalk  <-- THIS IS YOUR KEY