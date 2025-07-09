from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")

model.export(
    format="tflite",
    int8=True,
    imgsz=256,               # or whatever size you trained with
    data="dataset.yaml",     # points to your coco8-pose dataset

)

