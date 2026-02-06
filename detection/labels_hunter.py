from transformers import AutoModelForSemanticSegmentation

def find_hidden_labels():
    model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
    
    print(f"🔍 Scanning {len(model.config.id2label)} labels...")
    print("-" * 30)
    
    # Print everything so we can control-F locally
    for id, label in model.config.id2label.items():
        print(f"{id}: {label}")

if __name__ == "__main__":
    find_hidden_labels()