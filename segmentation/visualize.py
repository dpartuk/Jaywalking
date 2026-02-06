import matplotlib.pyplot as plt

def visualize_prediction(idx):
    # Load model state
    model.load_state_dict(torch.load("best_segformer_crosswalk.pt", weights_only=True))
    model.eval()
    model.to(DEVICE)
    
    # Get item
    dataset = val_dataset # or test_dataset
    item = dataset[idx]
    pixel_values = item["pixel_values"].unsqueeze(0).to(DEVICE)
    label = item["labels"].numpy()
    
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        upsampled_logits = torch.nn.functional.interpolate(
            outputs.logits, 
            size=label.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        pred_mask = upsampled_logits.argmax(dim=1).cpu().numpy()[0]

    # Plot
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original Image (Inverse Normalize needed for perfect visuals, but this works for rough view)
    img_vis = pixel_values.cpu().numpy()[0].transpose(1, 2, 0)
    img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min()) # Simple min-max norm
    
    ax[0].imshow(img_vis)
    ax[0].set_title("Input Image")
    
    ax[1].imshow(label, cmap="gray")
    ax[1].set_title("Ground Truth")
    
    ax[2].imshow(pred_mask, cmap="gray")
    ax[2].set_title("Prediction")
    
    plt.show()

# Run visualization
visualize_prediction(5)