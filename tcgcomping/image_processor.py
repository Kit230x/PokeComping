"""Image decoding and card detection/normalization. Keeps geometry separate from OCR.

Scans are rarely perfect: the card may be off-center, jammed against an edge, rotated,
shadowed, or sitting on a background that is nearly the same brightness as its silver
border. No single detection method handles all of those, so this module runs several,
scores every candidate on how card-like it is, and keeps the best one.
"""
import cv2
import numpy as np

CARD_ASPECT = 63.0 / 88.0          # width / height of a standard trading card (~0.716)
MIN_AREA_FRACTION = 0.04           # a card smaller than this share of the scan isn't trusted
MIN_SCORE = 0.30                   # below this, fall back instead of warping a bad guess
WORK_SIDE = 800                    # detection runs on a downscaled copy for speed and noise


def decode_image(image_bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image. Try a JPG or PNG scan.")
    return image


def _order_points(points):
    """Order corners as top-left, top-right, bottom-right, bottom-left.

    Sorts by angle around the centroid, so it stays correct for cards rotated up to and past
    45 degrees (the usual sum/difference trick produces duplicate corners there).
    """
    pts = np.asarray(points, dtype="float32").reshape(-1, 2)
    c = pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]  # clockwise on screen
    return np.roll(pts, -int(np.argmin(pts.sum(axis=1))), axis=0)


# --------------------------------------------------------------------------- candidates

def _lab(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)


def _estimate_background(lab):
    """Background colour as a LAB triple.

    Prefers the flattest stretches of the image border: a scanner background is perfectly
    uniform, while a card edge that touches the border has gradients and print. That keeps a
    card jammed against two edges from being mistaken for the background. When no stretch is
    flat (textured mat, or a card filling the frame) it uses the most common border colour.
    """
    h, w = lab.shape[:2]
    s = max(3, int(min(h, w) * 0.02))
    strips = [lab[:s], lab[-s:], lab[:, :s].transpose(1, 0, 2), lab[:, -s:].transpose(1, 0, 2)]
    flat_pixels = []
    for strip in strips:
        # strip is (s, length, 3) for top/bottom and (s, length, 3) after transpose for sides
        n_chunks = 8
        length = strip.shape[1]
        for c in range(n_chunks):
            chunk = strip[:, c * length // n_chunks:(c + 1) * length // n_chunks].reshape(-1, 3)
            if len(chunk) and chunk.std(axis=0).max() < 5.0:
                flat_pixels.append(chunk)
    if flat_pixels:
        flat = np.concatenate(flat_pixels)
        q = (flat / 8).astype(np.int32)
        keys = q[:, 0] * 10000 + q[:, 1] * 100 + q[:, 2]
        vals, counts = np.unique(keys, return_counts=True)
        return np.median(flat[keys == vals[np.argmax(counts)]], axis=0)

    border = np.concatenate([
        lab[:s].reshape(-1, 3), lab[-s:].reshape(-1, 3),
        lab[:, :s].reshape(-1, 3), lab[:, -s:].reshape(-1, 3),
    ])
    q = (border / 8).astype(np.int32)
    keys = q[:, 0] * 10000 + q[:, 1] * 100 + q[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    best = vals[np.argmax(counts)]
    members = border[keys == best]
    return np.median(members, axis=0)


def _morph(mask, op, k):
    """Morphology that treats everything outside the image as background.

    OpenCV's default border handling lets a closing grow a region that ends within k/2 pixels
    of the image edge all the way to that edge, which pushed card boundaries off the card.
    """
    pad = k + 2
    padded = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = cv2.morphologyEx(padded, op, kernel)
    return out[pad:-pad, pad:-pad]


def _clean(mask):
    mask = _morph(mask, cv2.MORPH_OPEN, max(3, (min(mask.shape) // 80) | 1))
    mask = _morph(mask, cv2.MORPH_CLOSE, max(5, (min(mask.shape) // 30) | 1))
    return mask


def _masks(small):
    """Yield (name, binary mask) foreground guesses; the scorer decides which one is right."""
    lab = _lab(small)
    bg = _estimate_background(lab)
    dist = np.linalg.norm(lab - bg, axis=2)
    dist = cv2.GaussianBlur(dist, (0, 0), 1.2)

    # 1) Colour distance from the background, at several sensitivities. Low thresholds catch
    #    silver-on-white borders; high ones avoid swallowing soft shadows.
    for t in (7, 12, 20, 32):
        yield f"bg>{t}", _clean((dist > t).astype(np.uint8) * 255)

    # 2) Local texture: artwork and text are busy, scanner backgrounds are flat.
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (9, 9))
    var = cv2.blur(gray * gray, (9, 9)) - mean * mean
    tex = (var > 40).astype(np.uint8) * 255
    tex = _morph(tex, cv2.MORPH_CLOSE, 25)
    yield "texture", _clean(tex)

    # 3) Edge map, closed into a region (handles backgrounds that match the border colour).
    blur = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    for lo, hi in ((20, 60), (40, 120)):
        edges = cv2.Canny(blur, lo, hi)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
        edges = _morph(edges, cv2.MORPH_CLOSE, 15)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(edges)
        cv2.drawContours(filled, cnts, -1, 255, -1)
        yield f"edges{lo}", _clean(filled)


def _hull_quad(cnt):
    """Four-corner polygon through the blob's convex hull, or None.

    Unlike a rotated rectangle this keeps perspective: a card photographed at a slight angle
    is a trapezoid, and forcing a rectangle onto it shifts every corner.
    """
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.08, 15):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype("float32")
        if len(approx) < 4:
            break
    return None


def _angles_ok(quad, tol=18.0):
    """True when every interior angle is within `tol` degrees of a right angle."""
    q = _order_points(quad)
    for i in range(4):
        a, b = q[i - 1] - q[i], q[(i + 1) % 4] - q[i]
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))
        if abs(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))) - 90.0) > tol:
            return False
    return True


def _quads_from_mask(mask):
    """Largest blob -> list of (quad, contour area, quad area, kind).

    Offers the rotated rectangle and, when the blob really is a four-sided shape, the
    perspective quadrilateral too. The scorer decides which one is the card.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area <= 0:
        return []
    out = []
    rect = cv2.minAreaRect(cnt)
    rect_area = rect[1][0] * rect[1][1]
    if rect_area > 0:
        out.append((cv2.boxPoints(rect).astype("float32"), float(area), float(rect_area), "rect"))
    # Only when a rectangle fits poorly. A tilted flat card fills its bounding rectangle
    # (about 0.97+); a trapezoid does not. Offering the hull quad for flat cards lets bumps
    # in the mask (shadows) pull it off the real corners.
    fill = area / rect_area if rect_area > 0 else 1.0
    quad = _hull_quad(cnt) if fill < 0.955 else None
    if quad is not None and _angles_ok(quad):
        qa = float(cv2.contourArea(quad))
        if qa > 0:
            out.append((quad, float(area), qa, "persp"))
    return out


# --------------------------------------------------------------------------- scoring

def _edge_contrast(lab, quad):
    """How different the pixels just inside each side are from just outside it.

    Sides that run along or past the image border can't be judged, so the median of the
    judgeable sides is used. Returns a value in [0, 1].
    """
    h, w = lab.shape[:2]
    centre = quad.mean(axis=0)
    diffs = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        n = max(12, int(np.linalg.norm(b - a) / 4))
        t = np.linspace(0.15, 0.85, n)[:, None]
        pts = a + (b - a) * t
        normal = pts - centre
        normal /= (np.linalg.norm(normal, axis=1, keepdims=True) + 1e-6)
        step = max(3.0, 0.015 * min(h, w))
        inside, outside = pts - normal * step, pts + normal * step
        ok = (
            (outside[:, 0] >= 0) & (outside[:, 0] < w) & (outside[:, 1] >= 0) & (outside[:, 1] < h)
            & (inside[:, 0] >= 0) & (inside[:, 0] < w) & (inside[:, 1] >= 0) & (inside[:, 1] < h)
        )
        if ok.sum() < n * 0.5:
            continue
        ci = lab[inside[ok, 1].astype(int), inside[ok, 0].astype(int)]
        co = lab[outside[ok, 1].astype(int), outside[ok, 0].astype(int)]
        diffs.append(float(np.median(np.linalg.norm(ci - co, axis=1))))
    if not diffs:
        return 0.5   # nothing judgeable (card fills the frame): neutral
    c = float(np.median(diffs))
    return c / (c + 6.0)


def _score(quad, contour_area, rect_area, shape, lab):
    h, w = shape
    rect = _order_points(quad)
    # Mean of opposite sides, so a perspective trapezoid is judged by its average size.
    sw = float((np.linalg.norm(rect[1] - rect[0]) + np.linalg.norm(rect[2] - rect[3])) / 2)
    sh = float((np.linalg.norm(rect[3] - rect[0]) + np.linalg.norm(rect[2] - rect[1])) / 2)
    if min(sw, sh) < 1:
        return 0.0
    aspect = min(sw, sh) / max(sw, sh)
    aspect_score = float(np.exp(-((aspect - CARD_ASPECT) / 0.055) ** 2))
    rectangularity = min(1.0, contour_area / rect_area)
    area_fraction = min(1.0, rect_area / float(h * w))
    if area_fraction < MIN_AREA_FRACTION:
        return 0.0
    size_score = min(1.0, area_fraction / 0.30)  # bigger cards are easier to trust
    contrast = _edge_contrast(lab, quad)
    return aspect_score * (rectangularity ** 2) * (0.6 + 0.4 * size_score) * (0.35 + 0.65 * contrast)


def find_card_quad(image):
    """Return (quad in full-image pixels, score, method) for the best card-like candidate."""
    h, w = image.shape[:2]
    scale = WORK_SIDE / float(max(h, w))
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
    scale = small.shape[1] / float(w)
    lab = _lab(small)
    sh, sw = small.shape[:2]

    found = []  # (score, rect_area, quad, method)
    for name, mask in _masks(small):
        for quad, carea, rarea, kind in _quads_from_mask(mask):
            s = _score(quad, carea, rarea, (sh, sw), lab)
            if kind == "persp":
                s *= 0.985   # tie goes to the plain rectangle; a trapezoid must earn it
            if s > 0:
                found.append((s, rarea, quad, name if kind == "rect" else name + "+persp"))

    best = (None, 0.0, "")
    if found:
        # Several methods usually agree on "a card". Take the best-scoring one, then snap
        # its sides onto the true card edge (the masks are only accurate to a few pixels).
        s, _, quad, name = max(found, key=lambda f: f[0])
        if _is_tight_coloured_crop(small, lab, quad):
            # The image is the card itself and its border is vivid (yellow, etc.): the
            # "background" the masks saw is really the card border, so use the whole frame.
            fh, fw = small.shape[:2]
            quad = np.array([[0, 0], [fw - 1, 0], [fw - 1, fh - 1], [0, fh - 1]], dtype="float32")
            name = "tight-crop"
        else:
            quad, name = _prefer_outer_candidate(small, lab, found, (s, quad, name))
            quad = _refine_quad(small, quad)
        best = (quad / scale, s, name)

    # The whole frame is a candidate too: already-cropped scans have no background to find.
    frame = np.array([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]], dtype="float32")
    fa = min(sw, sh) / float(max(sw, sh))
    frame_score = float(np.exp(-((fa - CARD_ASPECT) / 0.055) ** 2)) * 0.55  # prior: detected beats frame
    if frame_score > best[1]:
        best = (frame / scale, frame_score, "full-frame")
    return best


def _is_tight_coloured_crop(small, lab, quad):
    """True when the scan is a tight crop of a card whose border is a vivid colour.

    Scanner backgrounds are white, gray, black or a muted mat. If the flat "background"
    along the image border is strongly saturated, it is almost certainly the card's own
    border (yellow on most older sets), and the winning quad is just the artwork inside it.
    Requires the frame to be card-shaped and the quad to sit evenly inset from all four sides.
    """
    h, w = small.shape[:2]
    aspect = min(h, w) / float(max(h, w))
    if abs(aspect - CARD_ASPECT) > 0.05:
        return False
    bg_lab = _estimate_background(lab)
    bg_bgr = cv2.cvtColor(np.uint8([[bg_lab]]), cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2HSV)[0, 0]
    # Only yellow/gold counts (OpenCV hue 8-40). A navy or green scanner background is just as
    # saturated but is not a card border; treating it as one threw away the real card outline.
    if int(hsv[1]) < 110 or not 8 <= int(hsv[0]) <= 40:
        return False
    # A printed card border is one flat colour. Wood grain, a desk mat or a gradient has the
    # same hue but varies a lot, so a noisy ring means "background", not "card border".
    s = max(3, int(min(h, w) * 0.02))
    ring = np.concatenate([
        lab[:s].reshape(-1, 3), lab[-s:].reshape(-1, 3),
        lab[:, :s].reshape(-1, 3), lab[:, -s:].reshape(-1, 3),
    ])
    mad = np.median(np.abs(ring - np.median(ring, axis=0)), axis=0)   # robust to rounded corners
    if float(mad.max()) > 5.0:
        return False
    q = _order_points(quad)
    # A tight crop is axis-aligned; a tilted card on a coloured background is not.
    for i in range(4):
        d = q[(i + 1) % 4] - q[i]
        if min(abs(d[0]), abs(d[1])) > 0.02 * max(abs(d[0]), abs(d[1])):
            return False
    insets = [q[:, 0].min() / w, 1 - q[:, 0].max() / w, q[:, 1].min() / h, 1 - q[:, 1].max() / h]
    if min(insets) < 0.01 or max(insets) > 0.10:
        return False
    return max(insets) / min(insets) < 2.5


def _prefer_outer_candidate(small, lab, found, best):
    """Grow the winning quad to a larger candidate when the extra ring is part of the card.

    The highest-scoring candidate is often the artwork's inner edge: art-versus-border is a
    sharper edge than border-versus-background, so it scores well. Cards with a coloured
    border (yellow on most older sets) then lose that border and every OCR position shifts.
    A larger candidate that contains the winner is preferred only if the ring between them
    is NOT background coloured. A margin of scanner background (white strip beside the card)
    fails that test, so those larger boxes are still rejected.
    """
    best_score, best_quad, best_name = best
    h, w = small.shape[:2]
    bg = _estimate_background(lab)
    inner_poly = best_quad.astype(np.int32)
    inner_mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(inner_mask, inner_poly, 255)
    inner_area = float(inner_mask.sum() / 255)
    size = float(min(np.linalg.norm(best_quad[1] - best_quad[0]), np.linalg.norm(best_quad[3] - best_quad[0])))

    chosen = (best_quad, best_name)
    chosen_area = inner_area
    for score, rect_area, quad, name in sorted(found, key=lambda f: f[1]):
        if score < 0.6 * best_score or rect_area <= chosen_area * 1.01:
            continue
        # must contain the winner (corners inside, with a little tolerance)
        poly = quad.astype(np.float32).reshape(-1, 1, 2)
        tol = -0.02 * size
        if any(cv2.pointPolygonTest(poly, (float(px), float(py)), True) < tol for px, py in best_quad):
            continue
        outer_mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(outer_mask, quad.astype(np.int32), 255)
        ring = (outer_mask > 0) & (inner_mask == 0)
        if ring.sum() < 0.005 * h * w:
            continue
        ring_dist = np.linalg.norm(lab[ring] - bg, axis=1)
        if float(np.median(ring_dist)) > 14.0:       # non-background ring: part of the card
            chosen, chosen_area = (quad, name), float(outer_mask.sum() / 255)
    return chosen


def _refine_quad(small, quad, inner=0.035, outer=0.030):
    """Snap each side of `quad` onto the strongest nearby brightness edge.

    The coarse mask is only accurate to a few pixels, and a few pixels is enough to shift
    fixed OCR regions. For every side this samples brightness along the outward normal,
    finds the outermost strong step within a narrow window (so it lands on the card outline
    rather than the border/artwork boundary further inside, or a soft shadow outside), fits a
    robust line through the hits, and intersects neighbouring lines for the new corners.
    The refinement is discarded if it moves any corner too far.
    """
    h, w = small.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (0, 0), 1.0).astype(np.float32)
    quad = _order_points(quad)
    centre = quad.mean(axis=0)
    size = float(min(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[3] - quad[0])))
    lo, hi = -inner * size, outer * size
    offsets = np.arange(lo, hi + 0.5, 0.5, dtype=np.float32)

    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        n = 48
        t = np.linspace(0.12, 0.88, n)[:, None]
        pts = a + (b - a) * t
        normal = pts - centre
        normal = normal / (np.linalg.norm(normal, axis=1, keepdims=True) + 1e-6)
        # normal of the side itself (perpendicular to a->b), pointing away from the centre
        side = (b - a) / (np.linalg.norm(b - a) + 1e-6)
        perp = np.array([-side[1], side[0]], dtype=np.float32)
        if np.dot(perp, (a + b) / 2 - centre) < 0:
            perp = -perp
        hits = []
        for p in pts:
            xs = (p[0] + perp[0] * offsets).astype(np.float32)
            ys = (p[1] + perp[1] * offsets).astype(np.float32)
            if xs.min() < 0 or ys.min() < 0 or xs.max() > w - 1 or ys.max() > h - 1:
                continue  # side runs along the image border: nothing to snap to
            prof = cv2.remap(
                gray, xs.reshape(1, -1), ys.reshape(1, -1), cv2.INTER_LINEAR
            ).ravel()
            grad = np.abs(np.diff(prof))
            if grad.max() < 6:   # no real edge here (flat area)
                continue
            # Outermost strong edge, not just the strongest: the strongest is often the
            # border/artwork boundary a few percent inside the real card outline.
            strong = np.nonzero(grad >= max(6.0, 0.4 * grad.max()))[0]
            k = int(strong[-1])
            off = (offsets[k] + offsets[k + 1]) / 2.0
            hits.append(p + perp * off)
        if len(hits) >= 12:
            vx, vy, x0, y0 = cv2.fitLine(np.array(hits, dtype=np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            lines.append((np.array([x0, y0]), np.array([vx, vy])))
        else:
            # keep the original side as a line when it can't be refined
            d = (b - a) / (np.linalg.norm(b - a) + 1e-6)
            lines.append((a.astype(np.float64), d.astype(np.float64)))

    new = []
    for i in range(4):
        p1, d1 = lines[(i - 1) % 4]
        p2, d2 = lines[i]
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross) < 1e-3:
            return quad
        t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / cross
        new.append(p1 + d1 * t)
    new = np.array(new, dtype="float32")
    if np.max(np.linalg.norm(new - quad, axis=1)) > 0.06 * size:
        return quad
    return new


# --------------------------------------------------------------------------- warping

def _warp(image, quad, target_w, target_h):
    rect = _order_points(np.asarray(quad, dtype="float32"))
    side_w = float(np.linalg.norm(rect[1] - rect[0]))
    side_h = float(np.linalg.norm(rect[3] - rect[0]))
    if side_w > side_h:  # landscape: rotate the corner order so the card comes out portrait
        rect = np.roll(rect, -1, axis=0)
    dst = np.array(
        [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(
        image, matrix, (target_w, target_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def _inset(quad, fraction):
    """Pull the quad slightly inward so the warp excludes a sliver of background."""
    c = quad.mean(axis=0)
    return c + (quad - c) * (1.0 - fraction)


def normalize_card(image, target_w=1500, target_h=2100):
    """Find the card, straighten it, and return (aligned image, note).

    Falls back to an aspect-preserving canvas only when no candidate scores as card-like.
    """
    quad, score, method = find_card_quad(image)
    if quad is not None and score >= MIN_SCORE:
        warped = _warp(image, _inset(quad, 0.004), target_w, target_h)
        return warped, f"Card boundary detected ({method}, score {score:.2f})"

    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((target_h, target_w, 3), 245, dtype=np.uint8)
    y = (target_h - resized.shape[0]) // 2
    x = (target_w - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas, "Boundary not confidently detected; aspect-preserving fallback used"