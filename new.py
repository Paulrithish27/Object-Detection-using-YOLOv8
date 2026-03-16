from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
img = cv2.imread(r'cars_road.jpg')

resized_output = cv2.resize(img, (640, 480))

if img is None:
    print("Image not loaded")
else:
    results = model(resized_output)

    annotated_img = results[0].plot()

    cv2.imshow("YOLOv8 Object Detection", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Detection completed")
