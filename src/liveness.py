import cv2


def blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def check_liveness(image, face, blur_threshold=80.0, min_face_size=80):
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    width, height = x2 - x1, y2 - y1
    reasons = []
    passed = True

    if width < min_face_size or height < min_face_size:
        passed = False
        reasons.append("face_too_small")

    x1c, y1c = max(x1, 0), max(y1, 0)
    crop = image[y1c:y2, x1c:x2]
    if crop.size == 0:
        return False, ["invalid_crop"], 0.0

    score = blur_score(crop)
    if score < blur_threshold:
        passed = False
        reasons.append("low_texture_variance")

    return passed, reasons, float(score)
