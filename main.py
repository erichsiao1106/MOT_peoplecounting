import cv2
import time
from tracker.mot import (tracking_table_init_with_id,
                         tracking_table_init,
                         do_pairing,
                         remove_low_confidence,
                         none_type_checking)

CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

vc = cv2.VideoCapture("../crowd.mp4")
net = cv2.dnn.readNetFromDarknet("../yolov4-person.cfg", "../yolov4-person_best.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/256)

frame_count = 0
frame0_flag = 0  # 代表目前為最初始狀態，從frame_0開始
while cv2.waitKey(1) < 1:
    (grabbed, frame) = vc.read()
    if not grabbed:
        break
    # ----------------------------
    # inference - obj detection
    # ----------------------------
    start = time.time()
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    end = time.time()

    # ----------------------------
    # obj tracking
    # ----------------------------
    if frame0_flag == 0:
        # INITIALIZATION
        new = tracking_table_init_with_id(boxes)
        frame0_flag = 1
    else:
        # INITIALIZATION
        old = new
        new = tracking_table_init(boxes)

        # TRACKING
        do_pairing(new, old)  # pairing
        remove_low_confidence(new)  # removing
        none_type_checking(new)  # checking

    start_drawing = time.time()
    # 標上bounding box
    for i, (classid, score, box) in enumerate(zip(classes, scores, boxes)):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %.2f" % (class_names[classid[0]], score)

        # Drawing & Labeling
        color = (255, 0, 0)
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 標上ID
    for item in new:
        # Draw tracking IDs
        text = "ID={}, C={}" .format(item['id'], round(item['confidence'], 1))
        color = (0, 0, 255)
        cv2.putText(frame, text, (item['pos'][0] - 25, item['pos'][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    end_drawing = time.time()

    fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (end - start), (end_drawing - start_drawing) * 1000)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow("detections", frame)

    # for tracker debugging: save tracking table to csv file
    # cv2.imwrite("frame_ID_{}.jpg" .format(frame_count), frame)
    # df = pd.DataFrame(new)
    # df.to_csv("tracking_{}.csv" .format(frame_count))

    frame_count += 1
