"""
Flask REST API for Olympiad OMR Scanner
Runs on Cloud (Render.com, Railway, VPS) or Local PC.
"""

from flask import Flask, request, jsonify
import cv2
import base64
import numpy as np
import json
import os
from scanner_olympiad import scan_olympiad_omr

app = Flask(__name__)

# Allow maximum 32MB file upload
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "status": "online",
        "service": "Olympiad OMR Scanner API (30 questions, 50 points)",
        "endpoints": {
            "POST /api/scan": "Scan and evaluate an Olympiad OMR sheet (30 questions)"
        }
    })

@app.route('/api/scan', methods=['POST'])
def scan_endpoint():
    # 1. Validate file: accept 'image', 'file', or 'files'
    file = None
    if 'image' in request.files:
        file = request.files.get('image')
    elif 'file' in request.files:
        file = request.files.get('file')
    elif 'files' in request.files:
        files = request.files.getlist('files')
        if files:
            file = files[0]

    if file is None or file.filename == '':
        return jsonify({
            "success": False,
            "error": "Hech qanday rasm fayli yuklanmadi ('image' yoki 'file' maydoni kerak)"
        }), 400

    # 2. Get answer key if provided
    answer_key_raw = request.form.get('answer_key') or request.form.get('key') or request.form.get('keys')
    
    try:
        # Read file bytes directly into memory
        image_bytes = file.read()
        
        # Run OMR engine
        result = scan_olympiad_omr(image_bytes, answer_key_raw)
        
        # Convert annotated image to base64 JPEG
        annotated_img = result.pop("annotated_image", None)
        base64_img = None
        if annotated_img is not None:
            _, buffer = cv2.imencode('.jpg', annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            base64_img = base64.b64encode(buffer).decode('utf-8')
            
        result["annotated_image_base64"] = base64_img
        return jsonify(result), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Skanerlashda xatolik: {str(e)}"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print("=" * 60)
    print("Olympiad OMR Scanner API ishga tushmoqda...")
    print(f"Port: {port}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=False)
