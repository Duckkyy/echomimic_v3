# pip install retina-face
# we recommand tensorflow==2.15
import torch  # Import torch first to avoid TensorFlow/CUDA conflicts
from retinaface import RetinaFace
import sys
from PIL import Image, ImageDraw
import numpy as np


def draw_bbox(image_path, x1, y1, x2, y2, output_path=None, color=(255, 0, 0), width=3):
    """
    Draw a bounding box on an image and save it.
    
    Args:
        image_path: Path to input image
        x1, y1: Top-left corner
        x2, y2: Bottom-right corner
        output_path: Where to save (default: <name>_bbox.png)
        color: RGB tuple for box color (default: red)
        width: Line width in pixels
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    # Draw rectangle with specified width
    for i in range(width):
        draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline=color)
    
    # Save
    if output_path is None:
        base = image_path.rsplit('.', 1)[0]
        output_path = f"{base}_bbox.png"
    
    img.save(output_path)
    print(f"Saved: {output_path}")
    return output_path


def get_mask_coord(image_path):

    img = Image.open(image_path).convert("RGB")
    img = np.array(img)[:,:,::-1]
    if img is None:
        raise ValueError(f"Exception while loading {img}")

    height, width, _ = img.shape

    facial_areas = resp = RetinaFace.detect_faces(img) 
    if len(facial_areas) == 0:
        print (f'{image_path} has no face detected!')
        return None
    else:
        face = facial_areas['face_1']
        x,y,x2,y2 = face["facial_area"]
        
        return y,y2,x,x2,height,width

if __name__ == "__main__":
#   image_path = sys.argv[1]
    image_path = "datasets/custom_data/imgs/monalisa.jpg"
    y1, y2, x1, x2, height, width = get_mask_coord(image_path)
    print(f"Coords: x1={x1}, y1={y1}, x2={x2}, y2={y2}, h={height}, w={width}")
    
    # Draw bounding box on image
    draw_bbox(image_path, x1, y1, x2, y2)