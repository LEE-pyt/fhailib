import numpy as np
from pytorch_lightning import Trainer
from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.utils.callbacks import LoadModelCallback, get_callbacks
from pathlib import Path
def train_and_export_anomaly_detection_model(config_path: str,  export_mode: str):
    # 모델 및 데이터셋 설정
    config = get_configurable_parameters(config_path=config_path)
    config.optimization.export_mode = export_mode  # 내보내기 모드 설정

    # 데이터셋 로딩
    datamodule = get_datamodule(config)
    datamodule.prepare_data()
    datamodule.setup()


    # 모델 및 콜백 준비
    model = get_model(config)
    callbacks = get_callbacks(config)

    # 트레이너 생성 및 학습
    trainer = Trainer(**config.trainer, callbacks=callbacks)
    trainer.fit(model=model, datamodule=datamodule)

    # 모델 검증 및 테스트
    load_model_callback = LoadModelCallback(weights_path=trainer.checkpoint_callback.best_model_path)
    trainer.callbacks.insert(0, load_model_callback)
    test_results = trainer.test(model=model, datamodule=datamodule)
    output_path = Path(config["project"]["path"])
    openvino_model_path = output_path / "weights" / "openvino" / "model.bin"
    metadata = output_path / "weights" / "openvino" / "metadata.json"
    print(openvino_model_path.exists(), metadata.exists())

    return test_results

if __name__ == '__main__':
    config_path = r"E:\Python380\Lib\site-packages\anomalib\models\patchcore\config.yaml"
    export_mode = "torch"  # 'openvino', 'onnx', 'pt' 등으로 설정 가능
    test_results = train_and_export_anomaly_detection_model(config_path, export_mode)
