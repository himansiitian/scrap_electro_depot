import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
from PIL import Image, ImageEnhance
import io
import time
import os
import json
from datetime import datetime
import pytesseract
import easyocr
from typing import List, Dict, Tuple, Optional

class OneyComputerVisionScanner:
    def __init__(self, headless=True):
        """Initialize CV scanner with Selenium webdriver"""
        chrome_options = Options()
        if headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--start-maximized')
        chrome_options.add_argument('--force-device-scale-factor=1')
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, 10)
        
        # Configure OCR and template matching
        try:
            self.reader = easyocr.Reader(['fr', 'en'])
        except Exception as e:
            print(f"Error setting up EasyOCR: {str(e)}")
            self.reader = None
            
        self.oney_templates = []
        self.load_oney_templates()
        
        # Configure detection parameters
        self.min_confidence = 0.8
        self.distance_threshold = 5
        self.iou_threshold = 0.2
        self.min_logo_width = 15
        self.max_logo_width = 35
        self.min_logo_height = 20
        self.max_logo_height = 45
        
        # Initialize feature detector
        try:
            self.feature_detector = cv2.SIFT_create()
            self.detector_name = "SIFT"
        except:
            try:
                self.feature_detector = cv2.ORB_create()
                self.detector_name = "ORB"
            except:
                self.feature_detector = None
                self.detector_name = "None"
        print(f"Using {self.detector_name} feature detector")
        
    def detect_oney_logo_template_matching(self, image: np.ndarray, min_confidence: float = 0.8) -> List[Dict]:
        """
        Detect Oney logo using multiple detection methods with strict filtering:
        1. Feature matching (SIFT/ORB)
        2. Multiple template matching methods
        3. Multi-scale detection
        4. Clustering of nearby matches
        5. Size and confidence filtering
        """
        results = []
        
        if not self.oney_templates:
            self.load_oney_templates()
            if not self.oney_templates:
                print("Warning: No template images loaded!")
                return results
        
        # Convert to RGB for consistent matching
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        print(f"\nScanning with {len(self.oney_templates)} templates...")
        for template_info in self.oney_templates:
            template = template_info['gray']
            template_name = template_info['name']
            print(f"\nProcessing template: {template_name}")
            
            # 1. Feature Matching
            if self.feature_detector:
                try:
                    # Get keypoints and descriptors
                    kp1, des1 = self.feature_detector.detectAndCompute(template, None)
                    kp2, des2 = self.feature_detector.detectAndCompute(gray_image, None)
                    
                    if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
                        # Create matcher
                        if self.detector_name == "SIFT":
                            FLANN_INDEX_KDTREE = 1
                            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                            search_params = dict(checks=50)
                            matcher = cv2.FlannBasedMatcher(index_params, search_params)
                        else:
                            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                        
                        # Find matches
                        if self.detector_name == "SIFT":
                            matches = matcher.knnMatch(des1, des2, k=2)
                            # Apply ratio test
                            good_matches = []
                            for m, n in matches:
                                if m.distance < 0.7 * n.distance:
                                    good_matches.append(m)
                        else:
                            matches = matcher.match(des1, des2)
                            # Sort by distance
                            matches = sorted(matches, key=lambda x: x.distance)
                            good_matches = matches[:10]  # Take top 10 matches
                        
                        if len(good_matches) > 5:
                            print(f"Found {len(good_matches)} good feature matches")
                            
                            # Get matched keypoints
                            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                            
                            # Find homography
                            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                            if H is not None:
                                h, w = template.shape
                                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                                dst = cv2.perspectiveTransform(pts, H)
                                
                                # Convert to bounding box
                                x, y, w, h = cv2.boundingRect(np.int32(dst))
                                match_confidence = len(good_matches) / max(len(matches), 1)
                                
                                if match_confidence > min_confidence:
                                    results.append({
                                        'type': 'logo_feature_match',
                                        'template': template_name,
                                        'bbox': (x, y, w, h),
                                        'confidence': match_confidence,
                                        'method': self.detector_name
                                    })
                except Exception as e:
                    print(f"Feature matching error: {str(e)}")
            
            # 2. Template Matching with multiple methods
            # Use a more focused range of scales for the expected logo size
            scales = np.linspace(0.15, 0.4, 6)  # Focus on smaller scales since we know the logo is small
            
            methods = [
                (cv2.TM_CCOEFF_NORMED, "CCOEFF_NORMED", 0.6),
                (cv2.TM_CCORR_NORMED, "CCORR_NORMED", 0.65),
                (cv2.TM_SQDIFF_NORMED, "SQDIFF_NORMED", 0.4)
            ]
            
            for scale in scales:
                # Resize template
                width = int(template.shape[1] * scale)
                height = int(template.shape[0] * scale)
                
                if width < 10 or height < 10:  # Skip if too small
                    continue
                    
                if width > image.shape[1] or height > image.shape[0]:  # Skip if too large
                    continue
                
                resized_template = cv2.resize(template, (width, height))
                
                for method, method_name, base_threshold in methods:
                    try:
                        result = cv2.matchTemplate(gray_image, resized_template, method)
                        
                        # Handle different methods
                        if method == cv2.TM_SQDIFF_NORMED:
                            # For SQDIFF, smaller values are better
                            threshold = base_threshold
                            locations = np.where(result <= threshold)
                        else:
                            # For others, larger values are better
                            threshold = base_threshold
                            locations = np.where(result >= threshold)
                        
                        for pt in zip(*locations[::-1]):
                            confidence = 1.0 - result[pt[1], pt[0]] if method == cv2.TM_SQDIFF_NORMED else result[pt[1], pt[0]]
                            
                            results.append({
                                'type': 'logo_template_match',
                                'template': template_name,
                                'bbox': (int(pt[0]), int(pt[1]), width, height),
                                'confidence': float(confidence),
                                'scale': scale,
                                'method': method_name
                            })
                            
                            if confidence > 0.7:  # High confidence match
                                print(f"High confidence match ({confidence:.2f}) at scale {scale:.2f} using {method_name}")
                                
                    except Exception as e:
                        print(f"Error with {method_name} at scale {scale:.2f}: {str(e)}")
        
        # 1. First filter by minimum confidence
        results = [r for r in results if r['confidence'] >= self.min_confidence]
        
        # 2. Filter by size constraints
        valid_results = []
        for result in results:
            w, h = result['bbox'][2], result['bbox'][3]
            if (self.min_logo_width <= w <= self.max_logo_width and 
                self.min_logo_height <= h <= self.max_logo_height):
                valid_results.append(result)
        
        # 3. Group nearby matches
        if len(valid_results) > 0:
            # Group matches that are close to each other (within 5 pixels)
            grouped = []
            for match in valid_results:
                x, y = match['bbox'][0], match['bbox'][1]
                found_group = False
                for group in grouped:
                    gx, gy = group[0]['bbox'][0], group[0]['bbox'][1]
                    if abs(x - gx) <= self.distance_threshold and abs(y - gy) <= self.distance_threshold:
                        group.append(match)
                        found_group = True
                        break
                if not found_group:
                    grouped.append([match])
            
            # 4. Take only the best match from each group
            final_results = []
            for group in grouped:
                best_match = max(group, key=lambda x: x['confidence'])
                final_results.append(best_match)
            
            # 5. Sort by confidence and keep only the top match
            final_results = sorted(final_results, key=lambda x: x['confidence'], reverse=True)
            final_results = final_results[:1]  # Keep only the single best match
            
            return final_results
        
        return []

    def setup_selenium(self, headless=True):
        """Set up selenium webdriver"""
        chrome_options = Options()
        if headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--start-maximized')
        chrome_options.add_argument('--force-device-scale-factor=1')
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, 10)
        
    def setup_ocr_engines(self):
        """Set up OCR engines"""
        try:
            self.reader = easyocr.Reader(['fr', 'en'])
        except Exception as e:
            print(f"Error setting up EasyOCR: {str(e)}")
            self.reader = None
            
    def setup_template_matching(self):
        """Load Oney logo templates"""
        self.oney_templates = []
        template_dir = "oney_templates"
        
        if not os.path.exists(template_dir):
            print(f"Template directory {template_dir} not found!")
            return
            
        for template_file in os.listdir(template_dir):
            if template_file.endswith((".png", ".jpg")):
                try:
                    template_path = os.path.join(template_dir, template_file)
                    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                    if template is not None:
                        self.oney_templates.append({
                            'name': template_file,
                            'gray': template,
                            'color': cv2.imread(template_path)
                        })
                except Exception as e:
                    print(f"Error loading template {template_file}: {str(e)}")

    def handle_cookie_popup(self):
        """Try to handle common cookie consent popups"""
        try:
            # Common cookie accept button selectors
            selectors = [
                "button[id*='cookie-accept']",
                "button[id*='accept-cookies']",
                "button[class*='cookie-accept']",
                "button[class*='accept-cookies']",
                "#acceptCookies",
                ".accept-cookies",
                "[aria-label*='Accept cookies']"
            ]
            
            for selector in selectors:
                try:
                    button = self.driver.find_element(By.CSS_SELECTOR, selector)
                    button.click()
                    print("Cookie popup handled")
                    return
                except:
                    continue
        except:
            pass
    
    def capture_screenshot(self, url: str, wait_time: int = 5) -> Optional[np.ndarray]:
        """Capture webpage screenshot with section-based scrolling"""
        try:
            print("Loading page...")
            self.driver.get(url)
            time.sleep(wait_time)  # Longer wait for initial load
            
            # Handle cookie popup
            print("Handling cookie popup...")
            self.handle_cookie_popup()
            time.sleep(2)  # Wait for popup to be handled
            
            print("Scanning page in sections...")
            # Get total page height
            total_height = self.driver.execute_script("return Math.max(document.body.scrollHeight, document.body.offsetHeight, document.documentElement.clientHeight, document.documentElement.scrollHeight, document.documentElement.offsetHeight);")
            
            # Get viewport height
            viewport_height = self.driver.execute_script("return window.innerHeight")
            
            # Reset to top
            self.driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(1)
            
            # Capture screenshot
            screenshot = self.driver.get_screenshot_as_png()
            nparr = np.frombuffer(screenshot, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            print(f"Screenshot captured, size: {image.shape}")
            return image
            
        except Exception as e:
            print(f"Screenshot error: {str(e)}")
            return None

    def comprehensive_scan(self, url: str, save_debug_images: bool = True, min_confidence: float = 0.85) -> Dict:
        """Perform comprehensive CV scan of a webpage"""
        print(f"Scanning: {url}")
        
        # Capture screenshot
        image = self.capture_screenshot(url)
        
        if image is None:
            return {'url': url, 'error': 'Failed to capture screenshot'}
        
        results = {
            'url': url,
            'timestamp': datetime.now().isoformat(),
            'image_dimensions': image.shape,
            'detections': {
                'logo_matches': [],
                'text_detections': [],
                'widget_detections': []
            },
            'summary': {
                'oney_detected': False,
                'detection_count': 0,
                'confidence_scores': [],
                'template_matches': []
            }
        }
        
        # Run template matching with higher precision
        logo_results = self.detect_oney_logo_template_matching(image, min_confidence=min_confidence)
        
        results['detections']['logo_matches'] = logo_results
        
        # Calculate summary
        all_detections = logo_results
        results['summary']['detection_count'] = len(all_detections)
        results['summary']['oney_detected'] = len(all_detections) > 0
        
        # Extract confidence scores
        for detection in all_detections:
            if 'confidence' in detection:
                results['summary']['confidence_scores'].append(detection['confidence'])
        
        # Save debug images if requested
        if save_debug_images and all_detections:
            self.save_debug_image(image, all_detections, url)
        
        return results
    
    def save_debug_image(self, image: np.ndarray, detections: List[Dict], url: str):
        """Save debug image with detection bounding boxes"""
        debug_image = image.copy()
        
        # Save original image first
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        domain = url.split('/')[2].replace('.', '_')
        
        # Create debug directory if it doesn't exist
        debug_dir = "debug_images"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
            
        # Save original
        original_filename = f"original_{domain}_{timestamp}.jpg"
        cv2.imwrite(os.path.join(debug_dir, original_filename), image)
        
        # Draw detection boxes
        for detection in detections:
            if 'bbox' in detection:
                x, y, w, h = detection['bbox']
                confidence = detection.get('confidence', 0)
                method = detection.get('method', 'unknown')
                
                # Draw rectangle
                cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Draw label with confidence and method
                label = f"{confidence:.2f} {method}"
                cv2.putText(debug_image, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Save debug image
        debug_filename = f"debug_{domain}_{timestamp}.jpg"
        cv2.imwrite(os.path.join(debug_dir, debug_filename), debug_image)
        print(f"Saved debug images to {debug_dir}/{debug_filename}")

if __name__ == "__main__":
    # Initialize scanner
    scanner = OneyComputerVisionScanner()
    
    # Test URL
    url = "https://www.electrodepot.fr/"
    
    # Run scan
    try:
        results = scanner.comprehensive_scan(url)
        print("\nScan Results:")
        print(f"URL: {results['url']}")
        print(f"Detections: {results['summary']['detection_count']}")
        print(f"Oney detected: {results['summary']['oney_detected']}")
        if results['summary']['confidence_scores']:
            print(f"Confidence scores: {results['summary']['confidence_scores']}")
    except Exception as e:
        print(f"Error during scan: {str(e)}")
    finally:
        # Clean up
        scanner.driver.quit()
