#!/usr/bin/env python3
"""
Simple test script for anime face detection.
Usage:
    python test_face_detect.py <image_path>
    python test_face_detect.py datasets/custom_data/imgs/anime.png
"""

import sys
import os
from PIL import Image, ImageDraw

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.face_detect import get_mask_coord


def test_face_detection(image_path, output_path=None):
    """
    Test face detection on an image and visualize the result.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the annotated image (optional)
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return None
    
    print(f"\n{'='*50}")
    print(f"Testing face detection on: {image_path}")
    print(f"{'='*50}\n")
    
    # Run face detection
    try:
        y1, y2, x1, x2, h, w = get_mask_coord(image_path)
    except Exception as e:
        print(f"Error during detection: {e}")
        return None
    
    # Print results
    print(f"\n--- Detection Results ---")
    print(f"Image size: {w}x{h}")
    print(f"Face bounding box:")
    print(f"  x1={x1}, y1={y1}")
    print(f"  x2={x2}, y2={y2}")
    print(f"  width={x2-x1}, height={y2-y1}")
    print(f"  center=({(x1+x2)//2}, {(y1+y2)//2})")
    
    # Calculate percentages
    face_w_pct = (x2 - x1) / w * 100
    face_h_pct = (y2 - y1) / h * 100
    print(f"  Face covers: {face_w_pct:.1f}% width, {face_h_pct:.1f}% height")
    
    # Load image and draw bounding box
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    # Draw rectangle (red, 3px thick)
    for i in range(3):
        draw.rectangle(
            [x1-i, y1-i, x2+i, y2+i],
            outline=(255, 0, 0)
        )
    
    # Draw center cross
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    draw.line([(cx-10, cy), (cx+10, cy)], fill=(0, 255, 0), width=2)
    draw.line([(cx, cy-10), (cx, cy+10)], fill=(0, 255, 0), width=2)
    
    # Save or show
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_detected{ext}"
    
    img.save(output_path)
    print(f"\n✅ Saved annotated image to: {output_path}")
    
    return (y1, y2, x1, x2, h, w)


def test_all_images_in_folder(folder_path):
    """Test face detection on all images in a folder."""
    extensions = ('.png', '.jpg', '.jpeg', '.webp')
    images = [f for f in os.listdir(folder_path) 
              if f.lower().endswith(extensions) and not f.endswith('_detected.png')]
    
    print(f"\nFound {len(images)} images in {folder_path}")
    
    results = {}
    for img_name in sorted(images):
        img_path = os.path.join(folder_path, img_name)
        result = test_face_detection(img_path)
        results[img_name] = result
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    for name, res in results.items():
        if res:
            y1, y2, x1, x2, h, w = res
            print(f"✅ {name}: face at ({x1},{y1})-({x2},{y2})")
        else:
            print(f"❌ {name}: detection failed")
    
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default: test the anime image
        default_paths = [
            "datasets/custom_data/imgs/anime.png",
        ]
        
        print("No image path provided. Testing default images...")
        for path in default_paths:
            if os.path.exists(path):
                test_face_detection(path)
        
        print("\nUsage: python test_face_detect.py <image_path>")
        print("       python test_face_detect.py <folder_path>")
    else:
        path = sys.argv[1]
        
        if os.path.isdir(path):
            test_all_images_in_folder(path)
        else:
            test_face_detection(path)
