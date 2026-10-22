"""
Olympiad OMR Scanner Engine
Specifically calibrated for Uzbekistan Olympiad answer sheets (Informatika / Fan olimpiadalari):
- 30 questions, 3 sections:
  * 1-10: 0.9 points each (Max 9.0)
  * 11-20: 1.5 points each (Max 15.0)
  * 21-30: 2.6 points each (Max 26.0)
  * Total: 50.0 points
"""

import cv2
import numpy as np
import os
import json

def four_point_transform(image, pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def preprocess_and_warp(image):
    """
    CamScanner-like perspective transform:
    Detects sheet boundary if photographed on a desk/background,
    straightening any angle/tilt.
    If the image is already a cropped sheet, returns it directly.
    """
    orig_h, orig_w = image.shape[:2]
    # If the image itself is already vertical A4-like ratio
    current_aspect = orig_h / float(orig_w)
    
    # Resize temporarily for fast contour search
    scale = 800.0 / max(orig_h, orig_w)
    small = cv2.resize(image, (int(orig_w * scale), int(orig_h * scale)))
    
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(blurred, 40, 140)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    img_area = small.shape[0] * small.shape[1]
    for c in contours:
        area = cv2.contourArea(c)
        # Paper should take a significant portion of photo, but not be the entire frame
        if 0.40 * img_area < area < 0.98 * img_area:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                # Check aspect ratio of the 4-point quadrilateral
                pts = approx.reshape(4, 2)
                # order points
                rect = np.zeros((4, 2), dtype="float32")
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]
                
                w_approx = np.linalg.norm(rect[1] - rect[0])
                h_approx = np.linalg.norm(rect[3] - rect[0])
                if w_approx > 0:
                    quad_aspect = h_approx / float(w_approx)
                    # Vertical A4 sheet must have quad_aspect ~ 1.2 to 1.7
                    if 1.15 < quad_aspect < 1.75:
                        orig_pts = pts / scale
                        return four_point_transform(image, orig_pts)
                
    return image

def extract_columns_and_boxes(sheet_img):
    """
    Locates the 3 question columns (1-10, 11-20, 21-30).
    Uses high-precision layout proportions or contour bounding boxes.
    """
    # Normalize sheet to standard 1000px width for consistent layout
    std_w = 1000
    scale = std_w / float(sheet_img.shape[1])
    std_h = int(sheet_img.shape[0] * scale)
    std_img = cv2.resize(sheet_img, (std_w, std_h))
    
    # Try finding the 3 table boxes by contour
    gray = cv2.cvtColor(std_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    total_area = std_w * std_h
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / float(h)
        area = w * h
        # Columns have aspect ~0.55-0.70, area ~ 5-15% of page
        if 0.45 < aspect < 0.75 and 0.04 * total_area < area < 0.18 * total_area:
            # and y is in upper half of page
            if y < 0.3 * std_h:
                candidates.append((x, y, w, h))
                
    candidates = sorted(candidates, key=lambda b: b[0])
    
    # Check if 3 matching boxes exist
    found_cols = None
    if len(candidates) >= 3:
        for i in range(len(candidates) - 2):
            b1, b2, b3 = candidates[i], candidates[i+1], candidates[i+2]
            y_diff = max(abs(b1[1] - b2[1]), abs(b2[1] - b3[1]))
            h_diff = max(abs(b1[3] - b2[3]), abs(b2[3] - b3[3]))
            if y_diff < 40 and h_diff < 40 and (b2[0] - b1[0] > 180) and (b3[0] - b2[0] > 180):
                found_cols = [b1, b2, b3]
                break
                
    if found_cols is None:
        # Fallback to standard sheet proportions
        # Header is y: 0.125 to 0.480
        y1 = int(std_h * 0.125)
        y2 = int(std_h * 0.485)
        h = y2 - y1
        b1 = (int(std_w * 0.055), y1, int(std_w * 0.290), h)
        b2 = (int(std_w * 0.350), y1, int(std_w * 0.300), h)
        b3 = (int(std_w * 0.650), y1, int(std_w * 0.300), h)
        found_cols = [b1, b2, b3]

    return std_img, found_cols

def scan_column(std_img, col_box, q_start):
    """
    Scans 10 questions in a given column box using center-patch sampling.
    Returns detected answers, fill values, and bubble coordinates.
    """
    bx, by, bw, bh = col_box
    col_crop = std_img[by:by+bh, bx:bx+bw]
    col_gray = cv2.cvtColor(col_crop, cv2.COLOR_BGR2GRAY)
    
    # Header row is ~10% of column height
    header_h = int(bh * 0.10)
    row_h = (bh - header_h) / 10.0
    
    # Bubble center proportions relative to column width:
    # A: ~23%, B: ~35%, C: ~47%, D: ~59%
    bubble_x_ratios = [0.23, 0.35, 0.47, 0.59]
    letters = ['A', 'B', 'C', 'D']
    
    results = {}
    patch_rad = max(3, int(row_h * 0.18))
    
    for i in range(10):
        q_num = q_start + i
        y_center = int(header_h + (i + 0.5) * row_h)
        
        means = []
        bubble_coords = []
        
        for r_x in bubble_x_ratios:
            x_center = int(r_x * bw)
            patch = col_gray[
                max(0, y_center - patch_rad):min(bh, y_center + patch_rad + 1),
                max(0, x_center - patch_rad):min(bw, x_center + patch_rad + 1)
            ]
            mean_val = float(np.mean(patch))
            means.append(mean_val)
            
            # Absolute coordinates on std_img
            abs_cx = bx + x_center
            abs_cy = by + y_center
            bubble_coords.append((abs_cx, abs_cy))
            
        # The filled bubble has significantly lower grayscale value (darker ink)
        sorted_means = sorted(means)
        min_val = sorted_means[0]
        second_min = sorted_means[1]
        diff = second_min - min_val
        
        # Contrast check: marked bubble should be darker by at least 25 points, or below 120
        if diff >= 25.0 or (min_val < 110 and diff >= 18.0):
            marked_letter = letters[np.argmin(means)]
        else:
            marked_letter = None # Blank or ambiguous
            
        results[q_num] = {
            "marked": marked_letter,
            "contrast_diff": round(diff, 1),
            "means": [round(m, 1) for m in means],
            "bubble_coords": bubble_coords
        }
        
    return results

def draw_check_mark(img, center, size=9, color=(0, 160, 0), thickness=3):
    cx, cy = center
    pt1 = (cx - size, cy)
    pt2 = (cx - size // 3, cy + size)
    pt3 = (cx + size, cy - size)
    cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)
    cv2.line(img, pt2, pt3, color, thickness, cv2.LINE_AA)

def draw_cross_mark(img, center, size=8, color=(0, 0, 220), thickness=3):
    cx, cy = center
    cv2.line(img, (cx - size, cy - size), (cx + size, cy + size), color, thickness, cv2.LINE_AA)
    cv2.line(img, (cx - size, cy + size), (cx + size, cy - size), color, thickness, cv2.LINE_AA)

def parse_answer_key(raw_input):
    """
    Robust answer key parser:
    Accepts:
    - Continuous string without commas: 'CDBDCBACADCDABACDBABBCAABBBCC'
    - String with commas/spaces: 'C, D, B, D, ...'
    - Lowercase or uppercase: 'cdbdc...'
    - Cyrillic letters: 'А, В, С, Д' -> 'A, B, C, D'
    - JSON or dict
    - If user enters more than 30 letters, takes ONLY the first 30 (discards extras).
    """
    if not raw_input:
        return {}
    if isinstance(raw_input, dict):
        return {int(k): str(v).strip().upper() for k, v in raw_input.items() if str(k).isdigit() and int(k) <= 30}
    
    s = str(raw_input).strip().upper()
    # Kirill harflarini lotinga o'giramiz
    for cyr, lat in [('А', 'A'), ('В', 'B'), ('С', 'C'), ('Д', 'D')]:
        s = s.replace(cyr, lat)
        
    # JSON tekshiruvi
    if s.startswith('{') or s.startswith('['):
        try:
            data = json.loads(s)
            if isinstance(data, dict):
                return {int(k): str(v).strip().upper() for k, v in data.items() if str(k).isdigit() and int(k) <= 30}
            elif isinstance(data, list):
                return {i + 1: str(v).strip().upper() for i, v in enumerate(data[:30])}
        except Exception:
            pass
            
    import re
    # 1-A 2-B kabi raqamli format bo'lsa
    numbered = re.findall(r'([1-3]?[0-9])\s*[-.:)]?\s*([ABCD])', s)
    if len(numbered) >= 4:
        return {int(num): ans for num, ans in numbered if 1 <= int(num) <= 30}
        
    # Faqat A, B, C, D harflarini ajratib olamiz (virgulsiz yoki birga yozilgan bo'lsa ham)
    letters = re.findall(r'[ABCD]', s)
    # Faqat birinchi 30 tasini olamiz (ortiqchasi avtomatik tashlab yuboriladi)
    return {i + 1: ltr for i, ltr in enumerate(letters[:30])}

def scan_olympiad_omr(image_path_or_bytes, answer_key=None):
    """
    Main function to scan and evaluate an Olympiad OMR sheet.
    image_path_or_bytes: file path (str) or raw image bytes
    answer_key: dict, list, or string (with or without commas)
    """
    if isinstance(image_path_or_bytes, bytes):
        nparr = np.frombuffer(image_path_or_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(image_path_or_bytes)
        
    if image is None:
        raise ValueError("Could not decode or open image.")
        
    # Standardize answer_key format to dict {1: "C", ...}
    parsed_key = parse_answer_key(answer_key)

    # Step 1: Perspective warp if needed
    warped = preprocess_and_warp(image)
    
    # Step 2: Locate 3 question columns
    std_img, col_boxes = extract_columns_and_boxes(warped)
    
    # Step 3: Scan each column
    detected = {}
    for col_idx, (q_start, col_box) in enumerate([(1, col_boxes[0]), (11, col_boxes[1]), (21, col_boxes[2])]):
        col_res = scan_column(std_img, col_box, q_start)
        detected.update(col_res)
        
    # Step 4: Scoring according to official rules
    # 1-10: 0.9 points each
    # 11-20: 1.5 points each
    # 21-30: 2.6 points each
    weights = {}
    for q in range(1, 11):
        weights[q] = 0.9
    for q in range(11, 21):
        weights[q] = 1.5
    for q in range(21, 31):
        weights[q] = 2.6
        
    total_score = 0.0
    max_score = 50.0
    
    sec1_correct, sec1_earned = 0, 0.0
    sec2_correct, sec2_earned = 0, 0.0
    sec3_correct, sec3_earned = 0, 0.0
    
    annotated = std_img.copy()
    letters = ['A', 'B', 'C', 'D']
    questions_list = []
    
    for q in range(1, 31):
        q_data = detected.get(q, {})
        marked = q_data.get("marked")
        w = weights[q]
        
        correct = parsed_key.get(q)
        is_correct = False
        earned = 0.0
        
        if correct:
            if marked == correct:
                is_correct = True
                earned = w
                
        if is_correct:
            if q <= 10:
                sec1_correct += 1
                sec1_earned += earned
            elif q <= 20:
                sec2_correct += 1
                sec2_earned += earned
            else:
                sec3_correct += 1
                sec3_earned += earned
            total_score += earned
            
        # Draw checkmark ✔ or cross ✖ in the designated table columns
        col_idx = 0 if q <= 10 else (1 if q <= 20 else 2)
        bx, by, bw, bh = col_boxes[col_idx]
        header_h = int(bh * 0.10)
        row_h = (bh - header_h) / 10.0
        row_i = (q - 1) % 10
        cy = by + int(header_h + (row_i + 0.5) * row_h)
        check_x = bx + int(0.76 * bw)
        cross_x = bx + int(0.91 * bw)

        b_coords = q_data.get("bubble_coords", [])

        if parsed_key and correct:
            if is_correct:
                # To'g'ri bo'lsa: ✔ ustuniga yashil belgi qo'yamiz
                draw_check_mark(annotated, (check_x, cy), size=9, color=(0, 160, 0), thickness=3)
            else:
                # Xato bo'lsa: X ustuniga qizil xoch qo'yamiz (pufakcha ustiga qizil bo'yoq surilmaydi)
                draw_cross_mark(annotated, (cross_x, cy), size=8, color=(0, 0, 220), thickness=3)
                # To'g'ri javob doirachasiga yupqa yashil aylana chizamiz
                if correct in letters:
                    c_idx = letters.index(correct)
                    if c_idx < len(b_coords):
                        ccx, ccy = b_coords[c_idx]
                        cv2.circle(annotated, (ccx, ccy), 13, (0, 180, 0), 2)
        else:
            # Agar kalit kiritilmagan bo'lsa, o'quvchi belgilagan javobni ko'k hoshiya bilan belgilaymiz
            if marked and marked in letters:
                m_idx = letters.index(marked)
                if m_idx < len(b_coords):
                    mcx, mcy = b_coords[m_idx]
                    cv2.circle(annotated, (mcx, mcy), 13, (255, 120, 0), 2)

        questions_list.append({
            "question": q,
            "marked": marked,
            "correct_answer": correct,
            "is_correct": is_correct if correct else None,
            "score": round(earned, 2),
            "max_score": w
        })

    # Summary overlay on image
    if parsed_key:
        banner = f"Umumiy ball: {total_score:.1f} / {max_score:.1f} ({round((total_score/max_score)*100, 1)}%)"
        cv2.putText(annotated, banner, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 140, 0), 3)
    else:
        cv2.putText(annotated, "OMR Varoq skan qilindi", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200, 60, 0), 3)

    return {
        "success": True,
        "total_score": round(total_score, 2),
        "max_score": round(max_score, 2),
        "percentage": round((total_score / max_score) * 100, 1) if max_score > 0 else 0,
        "sections": {
            "section_1": {
                "name": "1-10-topshiriqlar",
                "ball_per_question": 0.9,
                "correct_count": sec1_correct,
                "total_count": 10,
                "earned_score": round(sec1_earned, 2),
                "max_score": 9.0
            },
            "section_2": {
                "name": "11-20-topshiriqlar",
                "ball_per_question": 1.5,
                "correct_count": sec2_correct,
                "total_count": 10,
                "earned_score": round(sec2_earned, 2),
                "max_score": 15.0
            },
            "section_3": {
                "name": "21-30-topshiriqlar",
                "ball_per_question": 2.6,
                "correct_count": sec3_correct,
                "total_count": 10,
                "earned_score": round(sec3_earned, 2),
                "max_score": 26.0
            }
        },
        "questions": questions_list,
        "annotated_image": annotated
    }

if __name__ == "__main__":
    import argparse
    import base64

    parser = argparse.ArgumentParser(description="Olympiad OMR Scanner")
    parser.add_argument("--image", "-i", type=str, default="test_sheet.png", help="Path to image file")
    parser.add_argument("--key", "-k", type=str, default="", help="Answer key (comma separated or JSON)")
    parser.add_argument("--json", action="store_true", help="Output result as JSON")
    parser.add_argument("--output", "-o", type=str, default="", help="Save annotated image to path")
    args = parser.parse_args()

    default_olympiad_key = {
        1: "C", 2: "D", 3: "B", 4: "D", 5: "C",
        6: "B", 7: "A", 8: "C", 9: "A", 10: "D",
        11: "C", 12: "D", 13: "A", 14: "B", 15: "A",
        16: "C", 17: "D", 18: "B", 19: "A", 20: "B",
        21: "B", 22: "C", 23: "A", 24: "A", 25: "B",
        26: "B", 27: "B", 28: "B", 29: "C", 30: "C"
    }

    selected_key = args.key if args.key else (default_olympiad_key if not args.json else None)

    if os.path.exists(args.image):
        res = scan_olympiad_omr(args.image, selected_key)
        
        if args.output:
            cv2.imwrite(args.output, res["annotated_image"])

        if args.json:
            # Convert annotated image to base64 if needed, remove numpy array
            annotated_img = res.pop("annotated_image", None)
            if args.output:
                res["annotated_image_path"] = args.output
            print(json.dumps(res, indent=2, ensure_ascii=False))
        else:
            cv2.imwrite("test_annotated_output.jpg", res["annotated_image"])
            print(f"Total: {res['total_score']} / {res['max_score']} ({res['percentage']}%)")
            for s_key, s_val in res["sections"].items():
                print(f"  {s_val['name']}: {s_val['correct_count']}/{s_val['total_count']} -> {s_val['earned_score']} ball")
    else:
        print(f"Fayl topilmadi: {args.image}")
