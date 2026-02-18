import os
import sys
import torch
from PIL import Image

# Add sum3_repo to path
repo_path = os.path.abspath("sum3_repo")
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

try:
    import sam3
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    
    print("Imports successful.")
    
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = os.path.join(sam3_root, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
    
    print(f"BPE Path: {bpe_path}")
    if not os.path.exists(bpe_path):
        print("ERROR: BPE path does not exist!")
    
    device = "cpu" # Default to CPU for verification
    
    print("Building model...")
    # NOTE: If sam3.pt is missing, this might try to download. 
    # We skip checkpoint loading for basic verification if necessary or let it fail gracefully.
    model = build_sam3_image_model(bpe_path=bpe_path, device=device)
    print("Model built successfully.")
    
    image_path = os.path.join(sam3_root, "assets", "images", "test_image.jpg")
    print(f"Image Path: {image_path}")
    if not os.path.exists(image_path):
        print("WARNING: Image path does not exist!")
    else:
        image = Image.open(image_path)
        processor = Sam3Processor(model, confidence_threshold=0.5)
        # inference_state = processor.set_image(image) # This might require correct model weights
        print("Processor initialized.")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
