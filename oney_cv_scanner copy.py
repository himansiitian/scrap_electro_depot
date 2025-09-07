import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support impor    def detect_oney_logo_template_matching(self, image: np.ndarray, min_confidence: float = 0.6) -> List[Dict]:
        """
        Detect Oney logo using multiple detection methods:
        1. Feature matching (SIFT/ORB)
        2. Multiple template matching methods
        3. Multi-scale detection
        """
        results = []
        
        if not self.oney_templates:
            self.load_oney_templates()
            if not self.oney_templates:
                print("Warning: No template images loaded!")
                return resultsd_conditions as EC
import requests
from PIL import Image, ImageEnhance
                methods = [
                    (cv2.TM_CCOEFF_NORMED, 'CCOEFF_NORMED', 0.8),
                    (cv2.TM_CCORR_NORMED, 'CCORR_NORMED', 0.85),
                    (cv2.TM_SQDIFF_NORMED, 'SQDIFF_NORMED', 0.2)  # Note: For SQDIFF, smaller values are better
                ]
                
                for method, method_name, base_threshold in methods:
                    try:
                        result = cv2.matchTemplate(gray_image, resized_template, method)
                        
                        # Adjust threshold based on method and scale
                        if method == cv2.TM_SQDIFF_NORMED:
                            # For SQDIFF, smaller values are better
                            current_confidence = base_threshold * (1.1 if scale < 0.5 else 1.0)
                            locations = np.where(result <= current_confidence)
                        else:
                            # For other methods, larger values are better
                            current_confidence = base_threshold * (0.9 if scale < 0.5 else 1.0)
                            locations = np.where(result >= current_confidence)
                            
                        if len(locations[0]) > 0:
                            print(f"Found {len(locations[0])} matches with {method_name} at scale {scale:.2f}")
                            
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
                    except Exception as e:
                        print(f"Error with {method_name}: {str(e)}")Image, ImageEnhance
import io
import time
import os
import json
from datetime import datetime
import pytesseract
import easyocr
from typing import List, Dict, Tuple, Optional

class OneyComputerVisionScanner:
    def __init__(self, headless=True, use_gpu=True):
        """Initialize the CV scanner"""
        self.setup_selenium(headless)
        self.setup_ocr_engines()
        self.setup_template_matching()
        
    def setup_selenium(self, headless):
        """Setup Selenium WebDriver for screen capture"""
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--allow-running-insecure-content")
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.set_window_size(1920, 1080)
        
    def handle_cookie_popup(self):
        """Handle cookie consent popup"""
        try:
            # Wait for cookie popup and try different possible selectors
            time.sleep(3)  # Increased wait time for popup to appear
            cookie_buttons = [
                "button#acceptAll",  # Common accept all button
                "button#onetrust-accept-btn-handler",  # OneTrust
                "button.accept-cookies",  # Generic class
                "button[contains(text(), 'Accepter')]",  # French text
                "button[contains(text(), 'Accept')]",  # English text
                "#cookie-law-info-bar button",  # Another common pattern
                ".cookie-consent button",  # Another common pattern
            ]
            
            for selector in cookie_buttons:
                try:
                    # Try clicking the button if it exists
                    button = self.driver.find_element("css selector", selector)
                    button.click()
                    print("Successfully handled cookie popup")
                    time.sleep(1)  # Wait for popup to disappear
                    return True
                except:
                    continue
                    
            print("Could not find cookie accept button with standard selectors")
            return False
        except Exception as e:
            print(f"Error handling cookie popup: {str(e)}")
            return False
    
    def setup_ocr_engines(self):
        """Setup OCR engines (Tesseract and EasyOCR)"""
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\Tesseract.exe'  # Windows path
        # Initialize EasyOCR (supports multiple languages)
        self.easyocr_reader = easyocr.Reader(['en', 'fr'])
        
        # Tesseract config for better accuracy
        self.tesseract_config = '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    
    def setup_template_matching(self):
        """Setup template images for Oney logo detection"""
        # In a real implementation, you'd have actual Oney logo templates
        self.oney_templates = []
        # This would load actual Oney logo images
        # self.load_oney_templates()
    
    def load_oney_templates(self, template_dir="oney_templates"):
        """Load Oney logo templates for matching"""
        if not os.path.exists(template_dir):
            print(f"Template directory {template_dir} not found. Creating sample templates...")
            self.create_sample_templates(template_dir)
        
        for filename in os.listdir(template_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                template_path = os.path.join(template_dir, filename)
                template = cv2.imread(template_path, cv2.IMREAD_COLOR)
                if template is not None:
                    self.oney_templates.append({
                        'name': filename,
                        'template': template,
                        'gray': cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                    })
    
    def create_sample_templates(self, template_dir):
        """Create sample Oney logo templates (for demo purposes)"""
        os.makedirs(template_dir, exist_ok=True)
        
        # Create a simple text-based Oney logo template
        img = np.ones((50, 150, 3), dtype=np.uint8) * 255
        cv2.putText(img, 'ONEY', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 102, 204), 2)
        cv2.imwrite(os.path.join(template_dir, 'oney_logo.png'), img)
        
        print(f"Sample template created in {template_dir}")
    
    def capture_screenshot(self, url: str, wait_time: int = 5) -> np.ndarray:
        """Capture screenshot of the webpage with full-page scanning"""
        try:
            print("Loading page...")
            self.driver.get(url)
            time.sleep(wait_time)  # Wait for page to load
            
            # Handle cookie popup
            print("Handling cookie popup...")
            self.handle_cookie_popup()
            time.sleep(2)  # Wait for popup to be handled

            print("Scanning page in sections...")
            # Get total page height
            total_height = self.driver.execute_script("return Math.max( document.body.scrollHeight, document.body.offsetHeight, document.documentElement.clientHeight, document.documentElement.scrollHeight, document.documentElement.offsetHeight );")
            
            # Get viewport height
            viewport_height = self.driver.execute_script("return window.innerHeight")
            
            # Initialize full page image
            all_sections = []
            current_position = 0
            overlap = 100  # Pixels of overlap between sections
            scroll_pause_time = 2  # Increased wait time for content to load
            max_attempts = 3  # Number of attempts per section
            
            print(f"Total page height: {total_height}px")
            
            while current_position < total_height:
                # Scroll to position
                self.driver.execute_script(f"window.scrollTo(0, {current_position});")
                time.sleep(scroll_pause_time)  # Wait for content to settle
                
                # Capture current viewport
                screenshot = self.driver.get_screenshot_as_png()
                image = Image.open(io.BytesIO(screenshot))
                section = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Store each section
                all_sections.append(section)
                print(f"Captured section {len(all_sections)}: {section.shape}")
                
                # Move to next section
                current_position += (viewport_height - overlap)
                print(f"Scanning section at position {current_position}/{total_height}")
                
            # Stitch all sections together
            print("Stitching sections together...")
            if not all_sections:
                return None
                
            full_page = all_sections[0]
            for i in range(1, len(all_sections)):
                overlap_height = min(overlap, all_sections[i].shape[0])
                full_page = np.vstack((full_page[:-overlap_height], all_sections[i]))
                print(f"Stitched section {i+1}, current height: {full_page.shape[0]}px")
            
            print("Completed full page capture")
            return full_page
            
        except Exception as e:
            print(f"Error capturing screenshot: {e}")
            return None
            
            while scroll_attempts < max_attempts:
                # Scroll down in smaller increments
                current_position += window_height
                self.driver.execute_script(f"window.scrollTo(0, {current_position});")
                print(f"Scrolling... Position: {current_position}")
                time.sleep(scroll_pause_time)
                
                # Check if we've reached the bottom
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if current_position > new_height:
                    # We've reached the bottom
                    print("Reached the bottom of the page")
                    break
                    
                if new_height > last_height:
                    # Content has loaded, increasing page height
                    last_height = new_height
                    scroll_attempts = 0  # Reset attempts when we find new content
                else:
                    scroll_attempts += 1
                
            # Scroll back to top for consistent capture
            print("Scrolling back to top...")
            self.driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(2)  # Wait for any visual elements to settle
            
            # Take screenshot
            screenshot = self.driver.get_screenshot_as_png()
            
            # Convert to OpenCV format
            image = Image.open(io.BytesIO(screenshot))
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            return image_cv
        except Exception as e:
            print(f"Error capturing screenshot: {e}")
            return None
    
    def detect_oney_logo_template_matching(self, image: np.ndarray, min_confidence: float = 0.85) -> List[Dict]:
        """Detect Oney logo using multiple matching methods"""
        results = []
        
        if not self.oney_templates:
            self.load_oney_templates()
            if not self.oney_templates:
                print("Warning: No template images loaded!")
                return results
        
        # Convert to RGB for consistent matching
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        
        print(f"Scanning with {len(self.oney_templates)} templates...")
        for template_info in self.oney_templates:
            template = template_info['gray']
            template_name = template_info['name']
            print(f"\nMatching template: {template_name}")
            
            # 1. Try SIFT feature matching first
            try:
                # Detect keypoints and descriptors
                kp1, des1 = sift.detectAndCompute(template, None)
                kp2, des2 = sift.detectAndCompute(gray_image, None)
                
                if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
                    # FLANN matcher
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                    search_params = dict(checks=50)
                    flann = cv2.FlannBasedMatcher(index_params, search_params)
                    
                    matches = flann.knnMatch(des1, des2, k=2)
                    
                    # Apply Lowe's ratio test
                    good_matches = []
                    for m, n in matches:
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)
                            
                    print(f"Found {len(good_matches)} good feature matches")
                    
                    if len(good_matches) > 10:  # Minimum number of good matches
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
                            bbox = cv2.boundingRect(np.int32(dst))
                            results.append({
                                'type': 'logo_feature_match',
                                'template': template_name,
                                'bbox': bbox,
                                'confidence': len(good_matches) / len(matches),
                                'method': 'SIFT'
                            })
            except Exception as e:
                print(f"SIFT matching error: {str(e)}")
                
            print("Trying template matching methods...")
            
            # Multi-scale template matching with smaller scales for tiny logos
            scales = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Include smaller scales
            print(f"Template {template_name} original size: {template.shape}")
            
            for scale in scales:
                # Resize template
                width = int(template.shape[1] * scale)
                height = int(template.shape[0] * scale)
                
                if width < 10 or height < 10 or width > image.shape[1] or height > image.shape[0]:
                    continue
                    
                print(f"Trying scale {scale:.2f} - Size: {width}x{height}")
                
                resized_template = cv2.resize(template, (width, height))
                
                # Template matching with adaptive threshold for small logos
                result = cv2.matchTemplate(gray_image, resized_template, cv2.TM_CCOEFF_NORMED)
                # Use lower threshold for smaller scales
                current_confidence = min_confidence * (0.9 if scale < 0.5 else 1.0)
                locations = np.where(result >= current_confidence)
                if len(locations[0]) > 0:
                    print(f"Found {len(locations[0])} matches at scale {scale:.2f} with confidence >= {current_confidence:.2f}")
                
                for pt in zip(*locations[::-1]):
                    results.append({
                        'type': 'logo_template_match',
                        'template': template_name,
                        'scale': scale,
                        'confidence': float(result[pt[1], pt[0]]),
                        'bbox': (pt[0], pt[1], width, height),
                        'center': (pt[0] + width//2, pt[1] + height//2)
                    })
        
        return results
    
    def detect_oney_text_ocr(self, image: np.ndarray) -> List[Dict]:
        """Detect Oney text using OCR"""
        results = []
        
        # Method 1: EasyOCR
        try:
            ocr_results = self.easyocr_reader.readtext(image)
            
            for (bbox, text, confidence) in ocr_results:
                if 'oney' in text.lower() and confidence > 0.5:
                    # Convert bbox to standard format
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    x, y = int(min(x_coords)), int(min(y_coords))
                    w, h = int(max(x_coords) - min(x_coords)), int(max(y_coords) - min(y_coords))
                    
                    results.append({
                        'type': 'text_easyocr',
                        'text': text,
                        'confidence': confidence,
                        'bbox': (x, y, w, h),
                        'center': (x + w//2, y + h//2)
                    })
        except Exception as e:
            print(f"EasyOCR error: {e}")
        
        # Method 2: Tesseract OCR
        try:
            # Preprocess image for better OCR
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply different preprocessing techniques
            preprocessed_images = [
                gray,
                cv2.GaussianBlur(gray, (3, 3), 0),
                cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            ]
            
            for i, processed_img in enumerate(preprocessed_images):
                # Get OCR data with bounding boxes
                data = pytesseract.image_to_data(processed_img, config=self.tesseract_config, output_type=pytesseract.Output.DICT)
                
                for j in range(len(data['text'])):
                    text = data['text'][j].strip()
                    confidence = int(data['conf'][j])
                    
                    if text and 'oney' in text.lower() and confidence > 30:
                        x, y, w, h = data['left'][j], data['top'][j], data['width'][j], data['height'][j]
                        
                        results.append({
                            'type': f'text_tesseract_method_{i}',
                            'text': text,
                            'confidence': confidence,
                            'bbox': (x, y, w, h),
                            'center': (x + w//2, y + h//2)
                        })
        
        except Exception as e:
            print(f"Tesseract OCR error: {e}")
        
        return results
    
    def detect_payment_widgets(self, image: np.ndarray) -> List[Dict]:
        """Detect payment widgets using computer vision techniques"""
        results = []
        
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Method 1: Button detection using edge detection and contours
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter for button-like shapes (rectangular areas of reasonable size)
            if 500 < area < 50000:  # Typical button size range
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Buttons typically have aspect ratios between 1.5 and 6
                if 1.5 < aspect_ratio < 6:
                    # Extract region of interest
                    roi = image[y:y+h, x:x+w]
                    
                    # Check if this region contains payment-related text
                    roi_text_results = self.analyze_payment_text_in_roi(roi)
                    
                    if roi_text_results:
                        results.append({
                            'type': 'payment_widget_shape',
                            'bbox': (x, y, w, h),
                            'center': (x + w//2, y + h//2),
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'text_analysis': roi_text_results
                        })
        
        # Method 2: Color-based detection for Oney brand colors
        # Oney typically uses blue and orange colors
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([130, 255, 255])
        orange_lower = np.array([10, 100, 100])
        orange_upper = np.array([25, 255, 255])
        
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
        
        # Find regions with Oney brand colors
        for color, mask in [('blue', blue_mask), ('orange', orange_mask)]:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Significant colored area
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    results.append({
                        'type': f'brand_color_{color}',
                        'bbox': (x, y, w, h),
                        'center': (x + w//2, y + h//2),
                        'area': area,
                        'color': color
                    })
        
        return results
    
    def analyze_payment_text_in_roi(self, roi: np.ndarray) -> Optional[Dict]:
        """Analyze text in a region of interest for payment-related keywords"""
        payment_keywords = [
            'pay', 'payment', 'payer', 'paiement',
            'buy', 'acheter', 'achat',
            'finance', 'financing', 'financement',
            '3x', '4x', 'times', 'fois',
            'installment', 'mensualité',
            'credit', 'crédit',
            'oney'
        ]
        
        try:
            # OCR on the ROI
            text = pytesseract.image_to_string(roi, config='--psm 8').lower()
            
            found_keywords = [keyword for keyword in payment_keywords if keyword in text]
            
            if found_keywords:
                return {
                    'text': text.strip(),
                    'keywords_found': found_keywords,
                    'keyword_count': len(found_keywords)
                }
        except Exception as e:
            pass
        
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
        # Only include text results near logo matches
        text_results = self.detect_oney_text_ocr(image) if logo_results else []
        # Only include widget detections near confirmed template matches
        widget_results = self.detect_payment_widgets(image) if logo_results else []
        
        results['detections']['logo_matches'] = logo_results
        results['detections']['text_detections'] = text_results
        results['detections']['widget_detections'] = widget_results
        
        # Calculate summary
        all_detections = logo_results + text_results + widget_results
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
        os.makedirs("debug_images", exist_ok=True)
        
        # Save the full captured page without annotations
        original_filename = f"debug_images/original_{domain}_{timestamp}.jpg"
        print(f"Saving full page capture to: {original_filename}")
        cv2.imwrite(original_filename, image)
        
        colors = {
            'logo_template_match': (0, 255, 0),  # Green
            'text_easyocr': (255, 0, 0),  # Blue
            'text_tesseract': (0, 255, 255),  # Yellow
            'payment_widget_shape': (255, 0, 255),  # Magenta
            'brand_color_blue': (255, 100, 0),  # Blue-orange
            'brand_color_orange': (0, 165, 255)  # Orange
        }
        
        for detection in detections:
            if 'bbox' in detection:
                x, y, w, h = detection['bbox']
                color = colors.get(detection['type'], (128, 128, 128))
                
                # Draw rectangle
                cv2.rectangle(debug_image, (x, y), (x + w, y + h), color, 2)
                
                # Add label
                label = f"{detection['type']}"
                if 'confidence' in detection:
                    label += f" {detection['confidence']:.2f}"
                
                cv2.putText(debug_image, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Save debug image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        domain = url.split('/')[2].replace('.', '_')
        filename = f"debug_{domain}_{timestamp}.jpg"
        
        os.makedirs("debug_images", exist_ok=True)
        cv2.imwrite(f"debug_images/{filename}", debug_image)
        print(f"Debug image saved: debug_images/{filename}")
    
    def scan_multiple_pages(self, base_url: str, pages: List[str] = None) -> List[Dict]:
        """Scan multiple pages of a website"""
        if pages is None:
            pages = ['', '/produits', '/tv-video', '/electromenager', '/informatique', '/telephone']
        
        results = []
        
        for page in pages:
            url = f"{base_url.rstrip('/')}{page}"
            try:
                result = self.comprehensive_scan(url)
                results.append(result)
                print(f"✓ Scanned: {url} - Detections: {result['summary']['detection_count']}")
            except Exception as e:
                print(f"✗ Error scanning {url}: {e}")
                results.append({'url': url, 'error': str(e)})
            
            time.sleep(2)  # Be respectful to the server
        
        return results
    
    def generate_report(self, results: List[Dict], output_file: str = None) -> str:
        """Generate a comprehensive report of the scanning results"""
        total_pages = len(results)
        pages_with_detections = sum(1 for r in results if r.get('summary', {}).get('oney_detected', False))
        
        report = f"""
ONEY COMPUTER VISION SCANNING REPORT
=====================================
Scan Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Pages Scanned: {total_pages}
Pages with Oney Detections: {pages_with_detections}
Detection Rate: {(pages_with_detections/total_pages*100):.1f}%

DETAILED RESULTS:
"""
        
        for result in results:
            if 'error' in result:
                report += f"\n❌ {result['url']}: ERROR - {result['error']}\n"
                continue
            
            summary = result.get('summary', {})
            detections = result.get('detections', {})
            
            report += f"\n{'='*60}\n"
            report += f"URL: {result['url']}\n"
            report += f"Oney Detected: {'✅ YES' if summary.get('oney_detected') else '❌ NO'}\n"
            report += f"Total Detections: {summary.get('detection_count', 0)}\n"
            
            if summary.get('confidence_scores'):
                avg_confidence = np.mean(summary['confidence_scores'])
                report += f"Average Confidence: {avg_confidence:.2f}\n"
            
            # Detail each detection type
            for detection_type, detection_list in detections.items():
                if detection_list:
                    report += f"\n{detection_type.upper()}:\n"
                    for detection in detection_list:
                        report += f"  - Type: {detection.get('type', 'Unknown')}\n"
                        if 'text' in detection:
                            report += f"    Text: '{detection['text']}'\n"
                        if 'confidence' in detection:
                            report += f"    Confidence: {detection['confidence']:.2f}\n"
                        if 'bbox' in detection:
                            x, y, w, h = detection['bbox']
                            report += f"    Location: ({x}, {y}) Size: {w}x{h}\n"
                        report += "\n"
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Report saved to: {output_file}")
        
        return report
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'driver'):
            self.driver.quit()

if __name__ == "__main__":
    # Create an instance of the scanner
    scanner = OneyComputerVisionScanner(headless=False)
    try:
        print("Starting scanner...")
        print("Loading templates...")
        
        # Load Oney logo templates
        scanner.load_oney_templates()
        
        # Scan homepage
        url = "https://www.electrodepot.fr/"
        print(f"\nScanning homepage: {url}")
        
        # Run comprehensive scan
        result = scanner.comprehensive_scan(url)
        
        if 'error' not in result:
            # Generate and save report
            report = scanner.generate_report([result], "oney_cv_scan_report.txt")
            
            # Save raw results
            with open("oney_cv_scan_results.json", "w", encoding='utf-8') as f:
                json.dump([result], f, indent=2, ensure_ascii=False)
            
            print("\nScan complete! Check:")
            print("- oney_cv_scan_report.txt for detailed report")
            print("- oney_cv_scan_results.json for raw data")
            print("- debug_images/ folder for detection visualizations")
        else:
            print(f"\nError during scan: {result['error']}")
            
    finally:
        scanner.close()
