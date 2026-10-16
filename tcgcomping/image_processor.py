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
    """Order corners as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def normalize_card(image, target_w=1500, target_h=2100):
    """Attempt perspective normalization; fall back to aspect-preserving canvas.

    The output canvas is 2x larger than before (1500x2100) so the tiny collector
    number text keeps enough pixels for OCR.
    """
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

        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
        else:
            # Rounded card corners often produce 5-8 points; use a rotated rectangle instead.
            pts = cv2.boxPoints(cv2.minAreaRect(contour)).astype("float32")

        rect = _order_points(pts)
        side_w = float(np.linalg.norm(rect[1] - rect[0]))
        side_h = float(np.linalg.norm(rect[3] - rect[0]))
        if min(side_w, side_h) == 0:
            continue

        # A real card is 63:88 (about 0.716). Reject contours with the wrong shape,
        # such as the whole frame or a background object.
        aspect = min(side_w, side_h) / max(side_w, side_h)
        if not 0.66 <= aspect <= 0.78:
            continue

        # Landscape detection: rotate the corner order so the card comes out portrait.
        if side_w > side_h:
            rect = np.roll(rect, -1, axis=0)

        dst = np.array(
            [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
            dtype="float32",
        )
        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, matrix, (target_w, target_h), flags=cv2.INTER_CUBIC)
        return warped, "Card boundary detected and normalized"

    # Preserve aspect ratio instead of stretching arbitrary input into a card shape.
    scale = min(target_w / w, target_h / h)
    resized = cv2.resize(
        image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
    )
    canvas = np.full((target_h, target_w, 3), 245, dtype=np.uint8)
    y = (target_h - resized.shape[0]) // 2
    x = (target_w - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas, "Boundary not confidently detected; aspect-preserving fallback used"