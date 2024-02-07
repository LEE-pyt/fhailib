from matplotlib import pyplot as plt
from anomalib.data.utils import read_image
from anomalib.deploy import OpenVINOInferencer
import time

def model_init(model_path, metadata):
    inferencer = OpenVINOInferencer(path=model_path, metadata=metadata, device="GPU")
    inferencer.predict(image=image)
    return inferencer

def inference(image, inferencer):
    stime = time.time()
    predictions = inferencer.predict(image=image)
    etime = time.time()
    print((etime-stime)*1000,"ms")
    return predictions

image_path = "E:/Mytask/main/anomalib-main/datasets/MVTec/capsule/test/crack/001.png"
model_path = r"E:\Mytask\main\anomalib-main\results\patchcore\mvtec\capsule\run\weights\openvino\model.bin"
metadata = r"E:\Mytask\main\anomalib-main\results\patchcore\mvtec\capsule\run\weights\openvino\metadata.json"
image = read_image(path=image_path)

inferencer = model_init(model_path, metadata)
predictions = inference(image, inferencer)

images = [predictions.image, predictions.anomaly_map, predictions.heat_map, predictions.pred_mask, predictions.segmentations]
titles = ["Original Image", "Anomaly Map", "Heat Map", "Prediction Mask", "Segmentations","Segmentations"]
plt.figure(figsize=(len(images)*2, 5))
for i in range(len(images)):
    plt.subplot(1, len(images), i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
plt.show()
