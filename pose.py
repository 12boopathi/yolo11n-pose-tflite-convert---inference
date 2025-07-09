import cv2
import numpy as np
import tensorflow as tf
from ultralytics.utils.plotting import Annotator, colors

# COCO keypoint skeleton connection pairs (17 keypoints)
skeleton = [
    (5, 7), (7, 9),        # Left arm
    (6, 8), (8, 10),       # Right arm
    (11, 13), (13, 15),    # Left leg
    (12, 14), (14, 16),    # Right leg
    (5, 6),                # Shoulders
    (11, 12),              # Hips
    (5, 11), (6, 12),      # Body sides
    (0, 1), (1, 2), (2, 3), (3, 4),  # Face
    (0, 5), (0, 6)         # Neck to shoulders
]

# Model quantization parameters
input_scale = 0.003921568859368563
input_zero_point = -128
output_scale = 0.005685732699930668
output_zero_point = -119

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="yolo11n-pose_saved_model/yolo11n-pose_full_integer_quant.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_size = input_details[0]['shape'][1]

# IOU function for NMS
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union != 0 else 0

# Apply NMS
def apply_nms(detections, iou_thresh=0.4):
    detections = sorted(detections, key=lambda x: x[1][4], reverse=True)
    final_dets = []
    while detections:
        best = detections.pop(0)
        final_dets.append(best)
        detections = [d for d in detections if iou(best[0], d[0]) < iou_thresh]
    return final_dets

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

print("✅ Running inference... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig_h, orig_w = frame.shape[:2]

    img = cv2.resize(frame, (input_size, input_size))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0

    img_int8 = np.round(img_norm / input_scale + input_zero_point).astype(np.int8)
    input_tensor = np.expand_dims(img_int8, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0]
    out = output_scale * (output.astype(np.float32) + (-output_zero_point))
    preds = out.T

    # Collect all detections with score > threshold
    raw_detections = []
    for det in preds:
        if det[4] < 0.5 or len(det) < 5 + 17 * 3:
            continue

        # Convert YOLO bbox to [x1, y1, x2, y2] in image scale
        x, y, w, h = det[0], det[1], det[2], det[3]
        x1 = int((x - w / 2) * orig_w)
        y1 = int((y - h / 2) * orig_h)
        x2 = int((x + w / 2) * orig_w)
        y2 = int((y + h / 2) * orig_h)

        raw_detections.append(((x1, y1, x2, y2), det))

    # Apply NMS
    final_detections = apply_nms(raw_detections)

    annotator = Annotator(frame, line_width=2)
    for (box, det) in final_detections:
        x1, y1, x2, y2 = box
        annotator.box_label([x1, y1, x2, y2], label='person', color=colors(0))

        kpts = det[5:5 + 17 * 3].reshape(-1, 3)
        keypoints_xy = []

        for i, (kx, ky, kv) in enumerate(kpts):
            cx = int(kx * orig_w)
            cy = int(ky * orig_h)
            keypoints_xy.append((cx, cy, kv))
            if kv > 0.5:
                cv2.circle(frame, (cx, cy), 3, colors(i), -1)

        for i, j in skeleton:
            if keypoints_xy[i][2] > 0.5 and keypoints_xy[j][2] > 0.5:
                pt1 = (keypoints_xy[i][0], keypoints_xy[i][1])
                pt2 = (keypoints_xy[j][0], keypoints_xy[j][1])
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    # Show output
    cv2.imshow("YOLOv8 Pose - TFLite INT8", annotator.result())
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

