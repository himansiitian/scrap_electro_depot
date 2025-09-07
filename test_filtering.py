import json

def calculate_iou(box1, box2):
    """Calculate intersection over union between two bounding boxes"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection coordinates
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
        
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0:
        return 0.0
        
    return intersection_area / union_area

def filter_matches(matches, min_confidence=0.8, distance_threshold=5, iou_threshold=0.2,
                  min_logo_width=15, max_logo_width=35, min_logo_height=20, max_logo_height=45):
    """Filter matches using confidence, size, proximity and overlap criteria"""
    if not matches:
        return matches
    
    # First filter by confidence and size
    valid_matches = []
    for match in matches:
        w, h = match['bbox'][2], match['bbox'][3]
        # Check if logo size is within valid range
        is_valid_size = (min_logo_width <= w <= max_logo_width and 
                       min_logo_height <= h <= max_logo_height)
        if match['confidence'] >= min_confidence and is_valid_size:
            valid_matches.append(match)
    
    print(f"After size and confidence filtering: {len(valid_matches)} matches")
    
    # Sort by confidence
    valid_matches = sorted(valid_matches, key=lambda x: x['confidence'], reverse=True)
    
    # Group by proximity
    groups = []
    for match in valid_matches:
        x, y = match['bbox'][0], match['bbox'][1]
        found_group = False
        for group in groups:
            # Check if this match is close to any match in the group
            g_match = group[0]  # Compare with first match in group
            gx, gy = g_match['bbox'][0], g_match['bbox'][1]
            if abs(x - gx) <= distance_threshold and abs(y - gy) <= distance_threshold:
                group.append(match)
                found_group = True
                break
        if not found_group:
            groups.append([match])
    
    print(f"Number of distinct location groups: {len(groups)}")
    
    # Keep only the best match from each group
    kept_matches = []
    for group in groups:
        best_match = max(group, key=lambda x: x['confidence'])
        kept_matches.append(best_match)
    
    print(f"After proximity filtering: {len(kept_matches)} matches")
    
    # Final overlap check
    final_matches = []
    for match in kept_matches:
        should_keep = True
        for kept_match in final_matches:
            iou = calculate_iou(match['bbox'], kept_match['bbox'])
            if iou > iou_threshold:
                should_keep = False
                break
        if should_keep:
            final_matches.append(match)
    
    print(f"After overlap filtering: {len(final_matches)} matches")
    return final_matches

# Load the current results
with open('oney_cv_scan_results.json', 'r') as f:
    results = json.load(f)

# Get the matches
matches = results[0]['detections']['logo_matches']
print(f"Original matches: {len(matches)}")

# Apply filtering
filtered_matches = filter_matches(matches)

# Print the filtered matches
print("\nFiltered matches:")
for match in filtered_matches:
    print(f"- Location: {match['bbox']}, Confidence: {match['confidence']:.2f}")
