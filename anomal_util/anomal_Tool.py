from matplotlib import pyplot as plt
from anomalib.data.utils import read_image
from anomalib.deploy import OpenVINOInferencer, TorchInferencer
import time
from pytorch_lightning import Trainer
from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.utils.callbacks import LoadModelCallback, get_callbacks
from pathlib import Path

def openvino_inference(image_path, model_path, metadata_path):
    """
    OpenVINO 모델을 사용하여 주어진 이미지에 대한 추론을 수행하고 결과를 시각화합니다.

    Parameters:
    - image_path (str): 추론할 이미지 파일의 경로입니다.
    - model_path (str): OpenVINO 모델 파일(.bin)의 경로입니다.
    - metadata_path (str): 모델 메타데이터 파일(.json)의 경로입니다.

    사용 예시:
    image_path = "path/to/image.png"
    model_path = "path/to/model.bin"
    metadata_path = "path/to/metadata.json"
    openvino_inference(image_path, model_path, metadata_path)
    """
    image = read_image(path=image_path)
    inferencer = OpenVINOInferencer(path=model_path, metadata=metadata_path, device="GPU")
    predictions = inferencer.predict(image=image)

    images = [predictions.image, predictions.anomaly_map, predictions.heat_map, predictions.pred_mask, predictions.segmentations]
    titles = ["Original Image", "Anomaly Map", "Heat Map", "Prediction Mask", "Segmentations"]

    plt.figure(figsize=(len(images) * 2, 5))
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img)
        plt.title(titles[i])
    plt.show()

def torch_inference(image_path, model_path):
    """
    PyTorch 모델을 사용하여 주어진 이미지에 대한 추론을 수행하고 결과를 시각화합니다.

    Parameters:
    - image_path (str): 추론할 이미지 파일의 경로입니다.
    - model_path (str): PyTorch 모델 파일(.pt)의 경로입니다.

    사용 예시:
    image_path = "path/to/image.png"
    model_path = "path/to/model.pt"
    torch_inference(image_path, model_path)
    """
    image = read_image(path=image_path)
    inferencer = TorchInferencer(path=model_path, device="cpu")
    predictions = inferencer.predict(image=image)

    images = [predictions.image, predictions.anomaly_map, predictions.heat_map, predictions.pred_mask, predictions.segmentations]
    titles = ["Original Image", "Anomaly Map", "Heat Map", "Prediction Mask", "Segmentations"]

    plt.figure(figsize=(len(images) * 2, 5))
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img)
        plt.title(titles[i])
    plt.show()

def train_export(config_path, export_mode):
    """
    이상 탐지 모델을 학습시키고 지정된 형식으로 모델을 내보냅니다.

    Parameters:
    - config_path (str): 모델 및 학습 설정이 정의된 YAML 구성 파일의 경로입니다.
    - export_mode (str): 모델을 내보낼 형식('torch', 'onnx', 'openvino')입니다.

    사용 예시:
    config_path = "path/to/config.yaml"
    export_mode = "torch"
    train_and_export_anomaly_detection_model(config_path, export_mode)
    """
    config = get_configurable_parameters(config_path=config_path)
    config.optimization.export_mode = export_mode

    datamodule = get_datamodule(config)
    datamodule.prepare_data()
    datamodule.setup()

    model = get_model(config)
    callbacks = get_callbacks(config)

    trainer = Trainer(**config.trainer, callbacks=callbacks)
    trainer.fit(model=model, datamodule=datamodule)

    load_model_callback = LoadModelCallback(weights_path=trainer.checkpoint_callback.best_model_path)
    trainer.callbacks.insert(0, load_model_callback)
    test_results = trainer.test(model=model, datamodule=datamodule)

    output_path = Path(config.project.path)
    model_export_path = output_path / "weights" / export_mode
    print(f"Model exported to {model_export_path}")

    return test_results