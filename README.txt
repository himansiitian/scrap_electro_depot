# Run the installation script
bash install_requirements.sh

# Or manual installation
pip install opencv-python numpy selenium easyocr pyautogui pillow beautifulsoup4


Results :
Debug image will show the webpage with colored rectangles around detected elements:
Green boxes: Logo template matches
Blue boxes: Text detected by EasyOCR
Yellow boxes: Text detected by Tesseract
Magenta boxes: Payment widget shapes
Blue-orange/Orange boxes: Areas with Oney brand colors
