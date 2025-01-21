from roboflow import Roboflow
from config import ROBOFLOW_API_KEY

rf = Roboflow(api_key=ROBOFLOW_API_KEY)

project = rf.workspace("objectdetectv1-03u2h").project("object-detection-dataset-bmpr6")
version = project.version(2)
dataset = version.download("yolov11")