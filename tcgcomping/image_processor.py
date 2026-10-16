"""Image decoding and card normalization. Keeps geometry separate from OCR."""
import cv2
import numpy as np

def decode_image(image_bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image. Try a JPG or PNG scan.")
    return image

def _order_points(points):
    rect = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect

def normalize_card(image, target_w=750, target_h=1050):
    """Attempt perspective normalization; fall back to aspect-preserving canvas."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 35, 130)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:15]:
        if cv2.contourArea(contour) < h * w * 0.18:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            pts = _order_points(approx.reshape(4, 2))
            dst = np.array([[0, 0], [target_w-1, 0], [target_w-1, target_h-1], [0, target_h-1]], dtype="float32")
            matrix = cv2.getPerspectiveTransform(pts, dst)
            return cv2.warpPerspective(image, matrix, (target_w, target_h)), "Card boundary detected and normalized"

    # Preserve aspect ratio instead of stretching arbitrary input into a card shape.
    scale = min(target_w / w, target_h / h)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), 245, dtype=np.uint8)
    y = (target_h - resized.shape[0]) // 2
    x = (target_w - resized.shape[1]) // 2
    canvas[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
    return canvas, "Boundary not confidently detected; aspect-preserving fallback used"
