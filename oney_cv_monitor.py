import cv2
import numpy as np
import pyautogui
import time
from datetime import datetime
import threading
import queue
import json
import os
from typing import Dict, List, Tuple
import easyocr
import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk

class RealTimeOneyMonitor:
    def __init__(self):
        self.setup_ui()
        self.setup_cv_components()
        self.monitoring = False
        self.detection_history = []
        self.screenshot_queue = queue.Queue(maxsize=10)
        
    def setup_ui(self):
        """Setup the monitoring GUI"""
        self.root = tk.Tk()
        self.root.title("Oney Widget Real-time Monitor")
        self.root.geometry("800x600")
        
        # Control frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=10, fill='x')
        
        self.start_btn = ttk.Button(control_frame, text="Start Monitoring", command=self.start_monitoring)
        self.start_btn.pack(side='left', padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop Monitoring", command=self.stop_monitoring, state='disabled')
        self.stop_btn.pack(side='left', padx=5)
        
        self.screenshot_btn = ttk.Button(control_frame, text="Take Screenshot", command=self.manual_screenshot)
        self.screenshot_btn.pack(side='left', padx=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready to monitor")
        self.status_label = ttk.Label(control_frame, textvariable=self.status_var)
        self.status_label.pack(side='right', padx=10)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(self.root, text="Settings")
        settings_frame.pack(pady=5, fill='x', padx=10)
        
        # Monitor interval
        ttk.Label(settings_frame, text="Monitor Interval (seconds):").grid(row=0, column=0, padx=5, pady=5)
        self.interval_var = tk.StringVar(value="5")
        ttk.Entry(settings_frame, textvariable=self.interval_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        # OCR confidence threshold
        ttk.Label(settings_frame, text="OCR Confidence Threshold:").grid(row=0, column=2, padx=5, pady=5)
        self.confidence_var = tk.StringVar(value="0.5")
        ttk.Entry(settings_frame, textvariable=self.confidence_var, width=10).grid(row=0, column=3, padx=5, pady=5)
        
        # Detection area selection
        self.area_select_btn = ttk.Button(settings_frame, text="Select Monitoring Area", command=self.select_area)
        self.area_select_btn.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        
        self.full_screen_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Monitor Full Screen", variable=self.full_screen_var).grid(row=1, column=2, columnspan=2, padx=5, pady=5)
        
        # Results display
        results_frame = ttk.LabelFrame(self.root, text="Detection Results")
        results_frame.pack(pady=5, fill='both', expand=True, padx=10)
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, width=80)
        self.results_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Detection counter
        self.counter_var = tk.StringVar(value="Detections: 0")
        ttk.Label(results_frame, textvariable=self.counter_var).pack(pady=2)
        
        # Monitoring area coordinates
        self.monitor_area = None
        
    def setup_cv_components(self):
        """Setup computer vision components"""
        # Initialize OCR
        try:
            self.ocr_reader = easyocr.Reader(['en', 'fr'])
            self.log_message("EasyOCR initialized successfully")
        except Exception as e:
            self.log_message(f"OCR initialization failed: {e}")
            self.ocr_reader = None
        
        # Oney detection patterns
        self.oney_keywords = [
            'oney', 'ONEY', 'Oney',
            '3x sans frais', '4x sans frais',
            'paiement en plusieurs fois',
            'financement',
            'crédit instantané'
        ]
        
        # Color ranges for Oney brand detection (HSV)
        self.oney_colors = {
            'blue': (np.array([100, 50, 50]), np.array([130, 255, 255])),
            'orange': (np.array([10, 100, 100]), np.array([25, 255, 255]))
        }
    
    def select_area(self):
        """Allow user to select monitoring area by drawing a rectangle"""
        self.root.withdraw()  # Hide main window
        
        # Take a screenshot for area selection
        screenshot = pyautogui.screenshot()
        screenshot_array = np.array(screenshot)
        
        # Simple area selection (you might want to implement a more sophisticated selector)
        self.log_message("Click and drag to select monitoring area, press ESC to cancel")
        
        # For this demo, we'll use a simple input dialog
        area_dialog = tk.Toplevel()
        area_dialog.title("Select Area")
        area_dialog.geometry("300x200")
        
        tk.Label(area_dialog, text="Enter coordinates (x, y, width, height):").pack(pady=10)
        
        coord_frame = tk.Frame(area_dialog)
        coord_frame.pack(pady=10)
        
        tk.Label(coord_frame, text="X:").grid(row=0, column=0)
        x_entry = tk.Entry(coord_frame, width=8)
        x_entry.grid(row=0, column=1, padx=5)
        x_entry.insert(0, "100")
        
        tk.Label(coord_frame, text="Y:").grid(row=0, column=2)
        y_entry = tk.Entry(coord_frame, width=8)
        y_entry.grid(row=0, column=3, padx=5)
        y_entry.insert(0, "100")
        
        tk.Label(coord_frame, text="W:").grid(row=1, column=0)
        w_entry = tk.Entry(coord_frame, width=8)
        w_entry.grid(row=1, column=1, padx=5)
        w_entry.insert(0, "800")
        
        tk.Label(coord_frame, text="H:").grid(row=1, column=2)
        h_entry = tk.Entry(coord_frame, width=8)
        h_entry.grid(row=1, column=3, padx=5)
        h_entry.insert(0, "600")
        
        def set_area():
            try:
                x = int(x_entry.get())
                y = int(y_entry.get())
                w = int(w_entry.get())
                h = int(h_entry.get())
                self.monitor_area = (x, y, w, h)
                self.full_screen_var.set(False)
                self.log_message(f"Monitoring area set to: {self.monitor_area}")
                area_dialog.destroy()
                self.root.deiconify()
            except ValueError:
                tk.messagebox.showerror("Error", "Please enter valid coordinates")
        
        def cancel():
            area_dialog.destroy()
            self.root.deiconify()
        
        ttk.Button(area_dialog, text="Set Area", command=set_area).pack(pady=5)
        ttk.Button(area_dialog, text="Cancel", command=cancel).pack(pady=5)
    
    def log_message(self, message: str):
        """Log a message to the results display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        self.results_text.insert(tk.END, formatted_message)
        self.results_text.see(tk.END)
        self.root.update_idletasks()
    
    def take_screenshot(self) -> np.ndarray:
        """Take a screenshot of the monitoring area"""
        try:
            if self.full_screen_var.get() or self.monitor_area is None:
                # Full screen
                screenshot = pyautogui.screenshot()
            else:
                # Specific area
                x, y, w, h = self.monitor_area
                screenshot = pyautogui.screenshot(region=(x, y, w, h))
            
            # Convert to OpenCV format
            screenshot_array = np.array(screenshot)
            screenshot_cv = cv2.cvtColor(screenshot_array, cv2.COLOR_RGB2BGR)
            
            return screenshot_cv
        
        except Exception as e:
            self.log_message(f"Screenshot error: {e}")
            return None
    
    def detect_oney_text(self, image: np.ndarray) -> List[Dict]:
        """Detect Oney-related text using OCR"""
        detections = []
        
        if self.ocr_reader is None:
            return detections
        
        try:
            confidence_threshold = float(self.confidence_var.get())
            
            # Use EasyOCR for text detection
            results = self.ocr_reader.readtext(image)
            
            for (bbox, text, confidence) in results:
                if confidence < confidence_threshold:
                    continue
                
                # Check if text contains Oney-related keywords
                text_lower = text.lower()
                matched_keywords = [keyword for keyword in self.oney_keywords if keyword.lower() in text_lower]
                
                if matched_keywords:
                    # Convert bbox to standard format
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    x, y = int(min(x_coords)), int(min(y_coords))
                    w, h = int(max(x_coords) - min(x_coords)), int(max(y_coords) - min(y_coords))
                    
                    detections.append({
                        'type': 'text_detection',
                        'text': text,
                        'matched_keywords': matched_keywords,
                        'confidence': confidence,
                        'bbox': (x, y, w, h),
                        'timestamp': datetime.now().isoformat()
                    })
        
        except Exception as e:
            self.log_message(f"OCR detection error: {e}")
        
        return detections
    
    def detect_oney_colors(self, image: np.ndarray) -> List[Dict]:
        """Detect Oney brand colors"""
        detections = []
        
        try:
            # Convert to HSV for color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            for color_name, (lower, upper) in self.oney_colors.items():
                mask = cv2.inRange(hsv, lower, upper)
                
                # Find contours in the mask
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    
                    # Filter for significant areas
                    if area > 500:
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        detections.append({
                            'type': 'color_detection',
                            'color': color_name,
                            'area': area,
                            'bbox': (x, y, w, h),
                            'timestamp': datetime.now().isoformat()
                        })
        
        except Exception as e:
            self.log_message(f"Color detection error: {e}")
        
        return detections
    
    def detect_payment_buttons(self, image: np.ndarray) -> List[Dict]:
        """Detect button-like shapes that might be payment widgets"""
        detections = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter for button-sized areas
                if 1000 < area < 20000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Typical button aspect ratios
                    if 1.5 < aspect_ratio < 5:
                        # Extract region of interest
                        roi = image[y:y+h, x:x+w]
                        
                        # Quick text analysis of the button region
                        roi_text = ""
                        if self.ocr_reader:
                            try:
                                roi_results = self.ocr_reader.readtext(roi)
                                roi_text = " ".join([result[1] for result in roi_results if result[2] > 0.3])
                            except:
                                pass
                        
                        # Check if it's likely a payment button
                        payment_indicators = ['pay', 'buy', 'acheter', 'payer', 'commander', '€', 'eur']
                        if any(indicator in roi_text.lower() for indicator in payment_indicators):
                            detections.append({
                                'type': 'button_detection',
                                'text': roi_text,
                                'area': area,
                                'aspect_ratio': aspect_ratio,
                                'bbox': (x, y, w, h),
                                'timestamp': datetime.now().isoformat()
                            })
        
        except Exception as e:
            self.log_message(f"Button detection error: {e}")
        
        return detections
    
    def process_screenshot(self, image: np.ndarray) -> Dict:
        """Process a screenshot for Oney widget detection"""
        if image is None:
            return {'detections': [], 'summary': {'total': 0, 'oney_found': False}}
        
        # Run all detection methods
        text_detections = self.detect_oney_text(image)
        color_detections = self.detect_oney_colors(image)
        button_detections = self.detect_payment_buttons(image)
        
        all_detections = text_detections + color_detections + button_detections
        
        # Create summary
        summary = {
            'total': len(all_detections),
            'text_detections': len(text_detections),
            'color_detections': len(color_detections),
            'button_detections': len(button_detections),
            'oney_found': len(text_detections) > 0,  # Text detections are most reliable for Oney
            'timestamp': datetime.now().isoformat()
        }
        
        return {
            'detections': all_detections,
            'summary': summary,
            'image_shape': image.shape
        }
    
    def monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                interval = float(self.interval_var.get())
                
                # Take screenshot
                screenshot = self.take_screenshot()
                
                if screenshot is not None:
                    # Process the screenshot
                    result = self.process_screenshot(screenshot)
                    
                    # Update detection history
                    self.detection_history.append(result)
                    
                    # Keep only last 100 detections
                    if len(self.detection_history) > 100:
                        self.detection_history.pop(0)
                    
                    # Update UI
                    self.update_results_display(result)
                    
                    # Save screenshot if detections found
                    if result['summary']['oney_found']:
                        self.save_detection_screenshot(screenshot, result)
                
                time.sleep(interval)
                
            except Exception as e:
                self.log_message(f"Monitoring error: {e}")
                time.sleep(1)
    
    def update_results_display(self, result: Dict):
        """Update the results display with new detection results"""
        summary = result['summary']
        detections = result['detections']
        
        # Update counter
        total_detections = len([h for h in self.detection_history if h['summary']['oney_found']])
        self.counter_var.set(f"Detections: {total_detections}")
        
        # Update status
        if summary['oney_found']:
            self.status_var.set("🔴 ONEY DETECTED!")
            self.log_message(f"🚨 ONEY WIDGET DETECTED! ({summary['total']} detections)")
            
            # Log each detection
            for detection in detections:
                if detection['type'] == 'text_detection':
                    self.log_message(f"  📝 Text: '{detection['text']}' (confidence: {detection['confidence']:.2f})")
                elif detection['type'] == 'color_detection':
                    self.log_message(f"  🎨 Color: {detection['color']} (area: {detection['area']})")
                elif detection['type'] == 'button_detection':
                    self.log_message(f"  🔘 Button: '{detection['text']}' (area: {detection['area']})")
        else:
            self.status_var.set("🟢 Monitoring...")
            if summary['total'] > 0:
                self.log_message(f"No Oney widgets found ({summary['total']} other detections)")
    
    def save_detection_screenshot(self, image: np.ndarray, result: Dict):
        """Save screenshot when Oney detection occurs"""
        try:
            # Create screenshots directory
            os.makedirs("oney_detections", exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"oney_detections/oney_detected_{timestamp}.jpg"
            
            # Draw detection boxes on image
            annotated_image = image.copy()
            
            colors = {
                'text_detection': (0, 255, 0),     # Green
                'color_detection': (255, 0, 0),    # Blue
                'button_detection': (0, 255, 255)  # Yellow
            }
            
            for detection in result['detections']:
                if 'bbox' in detection:
                    x, y, w, h = detection['bbox']
                    color = colors.get(detection['type'], (128, 128, 128))
                    
                    # Draw rectangle
                    cv2.rectangle(annotated_image, (x, y), (x + w, y + h), color, 2)
                    
                    # Add label
                    label = detection['type']
                    if detection['type'] == 'text_detection':
                        label += f": {detection['text'][:20]}"
                    
                    cv2.putText(annotated_image, label, (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Save image
            cv2.imwrite(filename, annotated_image)
            
            # Also save detection data as JSON
            json_filename = filename.replace('.jpg', '.json')
            with open(json_filename, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            self.log_message(f"💾 Detection saved: {filename}")
            
        except Exception as e:
            self.log_message(f"Error saving detection: {e}")
    
    def start_monitoring(self):
        """Start the monitoring process"""
        if not self.monitoring:
            self.monitoring = True
            self.start_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
            self.monitor_thread.start()
            
            self.log_message("🚀 Monitoring started")
            self.status_var.set("🟢 Monitoring...")
    
    def stop_monitoring(self):
        """Stop the monitoring process"""
        if self.monitoring:
            self.monitoring = False
            self.start_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
            
            self.log_message("⏹️ Monitoring stopped")
            self.status_var.set("Ready to monitor")
    
    def manual_screenshot(self):
        """Take a manual screenshot and analyze it"""
        self.log_message("📸 Taking manual screenshot...")
        
        screenshot = self.take_screenshot()
        if screenshot is not None:
            result = self.process_screenshot(screenshot)
            self.update_results_display(result)
            
            if result['summary']['oney_found']:
                self.save_detection_screenshot(screenshot, result)
            else:
                self.log_message("No Oney widgets detected in manual screenshot")
    
    def export_history(self):
        """Export detection history to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"oney_detection_history_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(self.detection_history, f, indent=2, ensure_ascii=False)
            
            self.log_message(f"📊 History exported: {filename}")
            
        except Exception as e:
            self.log_message(f"Export error: {e}")
    
    def run(self):
        """Run the monitoring application"""
        # Add export button
        export_btn = ttk.Button(self.root, text="Export History", command=self.export_history)
        export_btn.pack(pady=5)
        
        self.log_message("🎯 Oney Widget Monitor Ready")
        self.log_message("📋 Instructions:")
        self.log_message("   1. Optionally select a monitoring area")
        self.log_message("   2. Adjust settings if needed")
        self.log_message("   3. Click 'Start Monitoring'")
        self.log_message("   4. Browse to Electro Dépôt or any website")
        self.log_message("   5. Monitor will alert when Oney widgets are detected")
        
        # Start the GUI
        self.root.mainloop()

if __name__ == "__main__":
    # Requirements check
    required_packages = ['cv2', 'numpy', 'pyautogui', 'easyocr', 'PIL']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:", missing_packages)
        print("Install them with: pip install opencv-python numpy pyautogui easyocr Pillow")
        exit(1)
    
    # Create and run the monitor
    monitor = RealTimeOneyMonitor()
    monitor.run()