#!/bin/bash

# Oney Computer Vision Scanner - Installation Script
echo "🔧 Installing Oney Computer Vision Scanner Dependencies..."

# Create and activate conda environment
echo "📦 Creating conda environment..."
conda create -n oney_env python=3.8 -y

# Activate environment
echo "🔄 Activating conda environment..."
conda activate oney_env

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies with specific versions
echo "📥 Installing dependencies..."
pip install numpy==1.24.3
pip install pandas==1.5.3
pip install opencv-python==4.8.0.76
pip install pytesseract==0.3.10
pip install easyocr==1.7.1
pip install selenium

echo "✅ Installation complete!"
echo "To activate the environment, run: conda activate oney_env"



# Install core dependencies
echo "📚 Installing core Python packages..."
pip install Pillow
pip install requests
pip install beautifulsoup4
pip install lxml

# Install OCR engines
echo "🔍 Installing OCR engines..."
pip install pytesseract
pip install easyocr

# Install GUI dependencies
echo "🖥️ Installing GUI dependencies..."
pip install pyautogui

# Create requirements.txt for future reference
echo "📝 Creating requirements.txt..."
cat > requirements.txt << EOF
numpy=1.24.3
pandas=1.5.3
opencv-python=4.8.0.76
Pillow>=8.0.0
selenium>=4.0.0
requests>=2.25.0
beautifulsoup4>=4.9.0
lxml>=4.6.0
pytesseract>=0.3.10
easyocr>=1.7.1
pyautogui>=0.9.50
EOF

echo "✅ Requirements.txt created"

# Additional setup instructions
echo ""
echo "🎯 INSTALLATION COMPLETE!"
echo ""
echo "📋 ADDITIONAL SETUP REQUIRED:"
echo "1. Install Tesseract OCR:"
echo "   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
echo "   - macOS: brew install tesseract"
echo "   - Ubuntu: sudo apt install tesseract-ocr"
echo ""
echo "2. Download ChromeDriver:"
echo "   - Visit https://chromedriver.chromium.org/"
echo "   - Download version matching your Chrome browser"
echo "   - Add to PATH or place in same directory as script"
echo ""
echo "3. For the real-time monitor, you may need to:"
echo "   - Disable screen recording restrictions (macOS)"
echo "   - Grant accessibility permissions"
echo "   - Allow screenshot permissions"
echo ""
echo "🚀 USAGE:"
echo "1. Comprehensive Scanner: python oney_cv_scanner.py"
echo "2. Real-time Monitor: python oney_cv_monitor.py"
echo ""
echo "📁 OUTPUT FILES:"
echo "- debug_images/: Screenshots with detection annotations"
echo "- oney_detections/: Detected Oney widgets with metadata"
echo "- Reports: JSON and text reports of scanning results"