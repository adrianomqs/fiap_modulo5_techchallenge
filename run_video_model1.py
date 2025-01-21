from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11x.pt")

results = model.track("videos/video.mp4", save=True, show=True)