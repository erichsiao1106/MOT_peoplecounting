import cv2

class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
vc = cv2.VideoCapture("crowd.mp4")
# net = cv2.dnn.readNetFromDarknet("yolov4-person.cfg", "yolov4-person_best.weights")
net = cv2.dnn.readNetFromDarknet("../yolov4-person.cfg", "../yolov4-person_best.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/256)

frame_count = 0
while cv2.waitKey(1) < 1:
    (grabbed, frame) = vc.read()
    if not grabbed:
        break

    with open("boxes{}.txt" .format(frame_count), 'w') as f:
        classes, scores, boxes = model.detect(frame, 0.5, 0.4)
        for i, (classid, score, box) in enumerate(zip(classes, scores, boxes)):
            cx = box[0] + box[2] // 2
            cy = box[1] + box[3] // 2
            if 1000 >= cx >= 425 and 719 >= cy >= 300:
                f.write("{},{},{},{}\n" .format(box[0], box[1], box[2], box[3]))

    with open("frame{}.txt" .format(frame_count), 'w') as f:
        classes, scores, boxes = model.detect(frame, 0.5, 0.4)
        for i, (classid, score, box) in enumerate(zip(classes, scores, boxes)):
            cx = box[0] + box[2] // 2
            cy = box[1] + box[3] // 2
            if 1000 >= cx >= 425 and 719 >= cy >= 300:
                print("frame{} box: ( {} , {} )" .format(frame_count, cx, cy))
                label = "( {},{} )" .format(cx, cy)
                color = (255, 0, 0)
                cv2.rectangle(frame, box, color, 2)
                color = (0, 0, 255)
                cv2.putText(frame, label, (cx - 25, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                f.write("{},{}\n" .format(cx, cy))

    to_save = frame[300:719, 425:1100]
    cv2.imwrite("frame_{}.jpg" .format(frame_count), to_save)

    frame_count += 1
    # print("frame:", frame_count)
    if frame_count == 10:
        break
