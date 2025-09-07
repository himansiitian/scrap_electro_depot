import cv2
import numpy as np
import re
import os
from datetime import datetime

def parse_match(line):
    """Parse a match line from the report"""
    confidence_match = re.search(r'Confidence:\s*(0\.\d+)', line)
    location_match = re.search(r'Location:\s*\((\d+),\s*(\d+)\)', line)
    size_match = re.search(r'Size:\s*(\d+)x(\d+)', line)
    
    if confidence_match and location_match and size_match:
        confidence = float(confidence_match.group(1))
        x, y = int(location_match.group(1)), int(location_match.group(2))
        w, h = int(size_match.group(1)), int(size_match.group(2))
        return {
            'confidence': confidence,
            'bbox': (x, y, w, h)
        }
    return None

def draw_matches(image_path, matches, output_path):
    """Draw rectangles around matches and save the image"""
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Draw each match
    for match in matches:
        x, y, w, h = match['bbox']
        confidence = match['confidence']
        
        # Draw rectangle
        color = (0, 255, 0) if confidence >= 0.8 else (0, 165, 255)  # Green for high confidence, orange for lower
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        # Draw confidence score
        label = f"{confidence:.2f}"
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save the image
    cv2.imwrite(output_path, image)
    print(f"Debug image saved to: {output_path}")

# Read the report file
matches = []
with open('oney_cv_scan_report.txt', 'r') as f:
    lines = f.readlines()
    current_match = {}
    
    for line in lines:
        line = line.strip()
        if "Confidence:" in line:
            match = re.search(r'Confidence:\s*(0\.\d+)', line)
            if match:
                current_match['confidence'] = float(match.group(1))
        elif "Location:" in line and "Size:" in line:
            loc_match = re.search(r'\((\d+),\s*(\d+)\)', line)
            size_match = re.search(r'Size:\s*(\d+)x(\d+)', line)
            if loc_match and size_match:
                x = int(loc_match.group(1))
                y = int(loc_match.group(2))
                w = int(size_match.group(1))
                h = int(size_match.group(2))
                current_match['bbox'] = (x, y, w, h)
                if 'confidence' in current_match:
                    matches.append(current_match.copy())
                current_match = {}

print(f"\nFound {len(matches)} matches in the report")

# Group matches by location (within 5 pixels)
location_groups = {}
for match in matches:
    x, y = match['bbox'][0], match['bbox'][1]
    found_group = False
    for key in location_groups.keys():
        gx, gy = key
        if abs(x - gx) <= 5 and abs(y - gy) <= 5:
            location_groups[key].append(match)
            found_group = True
            break
    if not found_group:
        location_groups[(x, y)] = [match]

print(f"\nFound {len(location_groups)} distinct locations:")
for i, (loc, group) in enumerate(location_groups.items()):
    best_match = max(group, key=lambda x: x['confidence'])
    print(f"\nLocation Group {i+1}:")
    print(f"Center: ({loc[0]}, {loc[1]})")
    print(f"Best confidence: {best_match['confidence']:.2f}")
    print(f"Size: {best_match['bbox'][2]}x{best_match['bbox'][3]}")
    print(f"Matches in group: {len(group)}")

# Draw matches on the most recent debug image
debug_images = [f for f in os.listdir('debug_images') if f.startswith('debug_')]
if debug_images:
    latest_image = sorted(debug_images)[-1]
    input_path = f"debug_images/{latest_image}"
    output_path = f"debug_images/verified_{latest_image}"
    draw_matches(input_path, matches, output_path)
    print(f"\nVisualization saved to: {output_path}")

print("\nAnalysis of high-confidence matches (>=0.8):")
high_conf_matches = [m for m in matches if m['confidence'] >= 0.8]
if high_conf_matches:
    best_match = max(high_conf_matches, key=lambda x: x['confidence'])
    print(f"\nBest overall match:")
    print(f"Location: ({best_match['bbox'][0]}, {best_match['bbox'][1]})")
    print(f"Size: {best_match['bbox'][2]}x{best_match['bbox'][3]}")
    print(f"Confidence: {best_match['confidence']:.2f}")
else:
    print("No matches with confidence >= 0.8 found")
