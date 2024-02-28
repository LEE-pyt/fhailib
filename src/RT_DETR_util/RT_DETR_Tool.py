import torch
import json
import os
import onnxruntime as ort
from PIL import Image
from torchvision.transforms import ToTensor
import glob
from RT_DETR_util.tools import export_onnx
from RT_DETR_util.tools import train
from matplotlib import pyplot as plt
import requests
import zipfile
import shutil

def delete_unmatched_files(txt_files_path, images_path):
    """서로 매치되지 않는 이미지 파일과 레이블 파일을 삭제합니다."""
    txt_files = glob.glob(os.path.join(txt_files_path, '*.txt'))
    image_files = [txt.replace('labels', 'images').replace('.txt', '.jpg') for txt in txt_files]
    # 이미지 파일이 없는 .txt 파일 삭제
    for txt_file, image_file in zip(txt_files, image_files):
        if not os.path.exists(image_file):
            print(f"삭제: 이미지 파일이 없는 레이블 파일 {txt_file}")
            os.remove(txt_file)

    # 레이블 파일이 없는 이미지 파일 삭제
    for image_file in glob.glob(os.path.join(images_path, '*.jpg')):
        txt_file = image_file.replace('images', 'labels').replace('.jpg', '.txt')
        if not os.path.exists(txt_file):
            print(f"삭제: 레이블 파일이 없는 이미지 파일 {image_file}")
            os.remove(image_file)

def convert_labels_to_json(mscoco_category2name, txt_files_path, output_json_path, images_path):
    """
    .txt 파일들에서 레이블을 읽어 COCO 형식의 JSON 파일로 변환합니다.

    Parameters:
    - mscoco_category2name: 카테고리 ID와 이름의 매핑
    - txt_files_path: .txt 파일들이 있는 디렉토리 경로
    - output_json_path: 출력 JSON 파일의 경로

    예제 사용:
    mscoco_category2name = {
        0: 'person', 1: 'bicycle', # 이하 생략
    }
    txt_files_path = 'E:/Mytask/main/datasets/coco128/labels/train2017'
    output_json_path = 'E:/Mytask/main/datasets/coco128/train_annotations.json'

    convert_labels_to_json(mscoco_category2name, txt_files_path, output_json_path)
    """
    delete_unmatched_files(txt_files_path, images_path)
    coco_data = {"images": [], "annotations": [], "categories": []}
    image_id, annotation_id = 0, 0

    for category_id, category_name in mscoco_category2name.items():
        coco_data['categories'].append({"id": category_id, "name": category_name})

    for txt_file in glob.glob(os.path.join(txt_files_path, '*.txt')):
        image_path = txt_file.replace('labels', 'images').replace('.txt', '.jpg')
        with Image.open(image_path) as img:
            width, height = img.size

        coco_data['images'].append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": os.path.basename(txt_file).replace('.txt', '.jpg')
        })

        with open(txt_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                category_id, x_center, y_center, bbox_width, bbox_height = map(float, line.split())
                x_min = int((x_center - bbox_width / 2) * width)
                y_min = int((y_center - bbox_height / 2) * height)
                bbox_width = int(bbox_width * width)
                bbox_height = int(bbox_height * height)

                coco_data['annotations'].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(category_id),
                    "bbox": [x_min, y_min, bbox_width, bbox_height],
                    "area": bbox_width * bbox_height,
                    "iscrowd": 0
                })
                annotation_id += 1

        image_id += 1

    with open(output_json_path, 'w') as json_file:
        json.dump(coco_data, json_file)

def compare_model(model_path_1, model_path_2):
    """
    두 PyTorch 모델 파일의 상태 사전을 비교하여 구조가 같은지 확인하는 함수입니다.

    Parameters:
    - model_path_1: 첫 번째 모델 파일의 경로
    - model_path_2: 두 번째 모델 파일의 경로

    사용예시:
    model_path_1 = r'E:\Mytask\main\RT-DETR-main\output\rtdetr_r18vd_6x_coco\checkpoint0025.pth'
    model_path_2 = r'E:\Mytask\main\RT-DETR-main\model_converted.pt'
    compare_model_structures(model_path_1, model_path_2)
    """
    import torch

    # 상태 사전 불러오기
    state_dict_1 = torch.load(model_path_1, map_location=torch.device('cpu'))
    state_dict_2 = torch.load(model_path_2, map_location=torch.device('cpu'))

    # 상태 사전의 키 비교
    keys_1 = set(state_dict_1.keys())
    keys_2 = set(state_dict_2.keys())

    # 같은 키를 가지고 있는지 확인
    if keys_1 == keys_2:
        print("두 모델은 같은 구조를 가지고 있습니다.")
    else:
        print("두 모델은 다른 구조를 가지고 있습니다.")

        # 서로 다른 키가 무엇인지 출력
        only_in_1 = keys_1 - keys_2
        only_in_2 = keys_2 - keys_1
        if only_in_1:
            print("첫 번째 모델에만 존재하는 키들:", only_in_1)
        if only_in_2:
            print("두 번째 모델에만 존재하는 키들:", only_in_2)

def inferencer(image_path, model_path="model.onnx", thrh=0.5, mode=False):
    """
    ONNX 모델을 사용하여 이미지 또는 이미지 디렉토리 내의 모든 JPEG 이미지에 대한 추론을 수행하고,
    감지된 객체에 대해 경계 상자와 레이블로 주석을 달아 이미지를 표시하는 함수입니다.

    매개변수:
    - image_path (str): 처리할 단일 이미지 파일의 경로 또는 JPEG 이미지가 포함된 디렉토리 경로.
    - model_path (str): 추론에 사용할 ONNX 모델 파일의 경로. 기본값은 "model.onnx".
    - thrh (float): 객체 감지를 위한 신뢰도 임계값. 기본값은 0.5.
    - mode (str): 추론 모드 선택 ("True" 또는 "False"). 기본값은 "False".

    사용 예시:
    - 디렉토리 모드:
      inferencer("/path/to/your/image/directory", model_path="model.onnx", mode=True)
    - 단일 이미지 모드:
      inferencer("/path/to/your/image.jpg", model_path="model.onnx", thrh=0.5, mode=Fasle)
    """
    sess = ort.InferenceSession(model_path)
    mscoco_category2name = config_category2name()
    if mode == True:
        image_files = glob.glob(os.path.join(image_path, '*.jpg'))
    elif mode == False:
        image_files = [image_path]

    for image_file in image_files:
        im = Image.open(image_file).convert('RGB')
        im_resized = im.resize((640, 640))
        im_data = ToTensor()(im_resized).unsqueeze(0)  # 이미지 데이터를 텐서로 변환

        output = sess.run(
            output_names=None,
            input_feed={'images': im_data.numpy(), "orig_target_sizes": torch.tensor([[640, 640]]).numpy()},
        )

        labels, boxes, scores = output

        fig, ax = plt.subplots(1)
        ax.imshow(im_resized)

        for score, label, box in zip(scores[0], labels[0], boxes[0]):
            if score > thrh:
                label_name = mscoco_category2name.get(label-1, 'Unknown')  # 라벨 이름을 가져옴, 해당 라벨이 없으면 'Unknown'
                x1, y1, x2, y2 = box
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1, f'{label_name}: {score:.2f}', color='blue', fontsize=12, verticalalignment='top')

        plt.axis('off')
        plt.show()

def train_model(yml_path, model_path):
    """
    RT-DETR 모델을 지정된 YAML 설정 파일과 체크포인트를 사용하여 학습시키는 함수입니다. 이 함수는 RT-DETR 프로젝트의
    학습 설정과 체크포인트 파일을 이용하여 모델 학습을 실행합니다. 학습 과정을 시작하기 전에 설정 파일을 통해
    모든 학습 매개변수를 정의해야 합니다.

    Parameters:
    - yml_path (str): 학습 설정이 정의된 YAML 파일의 경로. 이 파일에는 학습률, 배치 크기, 에폭 수 등의 학습 설정이 포함됩니다.
    - model_path (str): 학습을 재개할 때 사용할 체크포인트 파일의 경로. 체크포인트 파일은 모델의 사전 학습된 가중치를 포함합니다.

    사용 예시:
    ```python
    yml_path = "E:/Mytask/main/RT-DETR-main/rtdetr_pytorch/configs/rtdetr/rtdetr_r18vd_6x_coco.yml"
    model_path = "E:/Mytask/main/RT-DETR-main/output/rtdetr_r18vd_6x_coco/checkpoint0053.pth"
    train_model(yml_path, model_path)
    ```

    이 함수를 사용하기 전에 RT-DETR 프로젝트의 학습 스크립트(train.py)가 올바른 경로에 위치해 있어야 하며,
    해당 스크립트는 이 함수 내에서 `train.main(args)`를 통해 호출됩니다. 사용자는 이 함수를 호출하기 전에
    RT-DETR 프로젝트의 환경 설정이 완료되어 있어야 합니다.
    """
    class Train_Args:
        config = yml_path
        resume = None
        tuning = model_path
        test_only = False
        amp = False


    args = Train_Args()
    train.main(args)

def Convert_Onnx(yml_path, model_path):

    """
    RT-DETR 모델을 ONNX 형식으로 변환하고 지정된 폴더에 저장하는 함수입니다. 이 함수는 RT-DETR 모델의
    체크포인트와 설정 파일을 기반으로 ONNX 모델을 생성합니다. 변환된 모델은 체크포인트가 위치한 폴더 내에
    'onnx_model'라는 하위 폴더에 저장됩니다. 이 폴더 이름은 체크포인트 파일명을 기반으로 한 ONNX 파일명으로 생성됩니다.

    Parameters:
    - yml_path (str): 변환 설정이 정의된 YAML 파일의 경로. 이 파일에는 모델 구조 및 변환에 필요한 설정 정보가 포함됩니다.
    - model_path (str): 변환할 모델 체크포인트 파일의 경로. 이 경로는 모델의 가중치와 변환할 모델의 정보를 포함합니다.

    사용 예시:
    ```python
    yml_path = "E:/Mytask/main/RT-DETR-main/rtdetr_pytorch/configs/rtdetr/rtdetr_r18vd_6x_coco.yml"
    model_path = "E:/Mytask/main/RT-DETR-main/output/rtdetr_r18vd_6x_coco/2024-02-06_12-39-59/best.pt"
    Convert_Onnx(yml_path, model_path)
    ```

    이 함수를 실행하기 전에는 RT-DETR 프로젝트의 ONNX 변환 스크립트가 올바른 경로에 위치하고, 필요한 모든 환경 설정이
    완료되어 있어야 합니다. 변환 프로세스는 `export_onnx.main(args)`를 호출하여 시작됩니다. 변환된 ONNX 파일은
    모델 체크포인트와 같은 디렉토리 내 'onnx_model' 폴더에 저장됩니다.
    """

    output_dir = os.path.join(os.path.dirname(model_path), "onnx_model")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    class Convert_Onnx_Args:
        def __init__(self):
            self.model_name = os.path.basename(model_path).split('.')[0]
            self.filename = f"{self.model_name}.onnx"
            self.config = yml_path
            self.resume = model_path
            self.file_name = os.path.join(output_dir, self.filename)
            self.check = False
            self.simplify = False

    args = Convert_Onnx_Args()
    export_onnx.main(args)

def download_file(url, save_path):
    """지정된 URL에서 파일을 다운로드하고 지정된 경로에 저장하는 함수"""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def move_and_cleanup_dataset(extract_to):
    """압축 해제된 데이터셋에서 필요한 작업을 수행하는 함수"""
    # train2017 폴더에서 상위 폴더로 파일 이동
    for category in ['images', 'labels']:
        src_folder = os.path.join(extract_to, 'coco128', category, 'train2017')
        dst_folder = os.path.join(extract_to, category)
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder, exist_ok=True)
        for file_name in os.listdir(src_folder):
            shutil.move(os.path.join(src_folder, file_name), dst_folder)

        # 원래의 train2017 폴더 삭제
        shutil.rmtree(src_folder)

    # coco128 폴더 밖으로 images와 labels 폴더 이동
    for category in ['images', 'labels']:
        src_folder = os.path.join(extract_to, 'coco128', category)
        dst_folder = os.path.join(extract_to, category)
        #shutil.move(src_folder, extract_to)

    # 더 이상 필요 없는 coco128 폴더 삭제
    shutil.rmtree(os.path.join(extract_to, 'coco128'))

def unzip_dataset(zip_path, extract_to):
    """다운로드한 zip 파일을 압축 해제하는 함수"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    # 압축 해제 후 필요한 파일 정리 및 이동
    move_and_cleanup_dataset(extract_to)

def download_dataset(dataset_path):
    """데이터셋 폴더의 존재 여부를 확인하고, 없으면 다운로드하는 함수"""
    if not os.path.exists(os.path.join(dataset_path, 'images')) or not os.path.exists(os.path.join(dataset_path, 'labels')):
        print(f"{dataset_path} 내 images 또는 labels 폴더가 존재하지 않습니다. 데이터셋을 다운로드합니다.")
        os.makedirs(dataset_path, exist_ok=True)  # 폴더 생성
        save_path = os.path.join(dataset_path, 'dataset.zip')
        download_file("https://ultralytics.com/assets/coco128.zip", save_path)
        print("다운로드 완료.")
        unzip_dataset(save_path, dataset_path)
        os.remove(save_path)
    else:
        print(f"{dataset_path} 내 images와 labels 폴더가 이미 존재합니다.")

def config_category2name():
    mscoco_category2name = {
        1: 'person',
        2: 'bicycle',
        3: 'car',
        4: 'motorcycle',
        5: 'airplane',
        6: 'bus',
        7: 'train',
        8: 'truck',
        9: 'boat',
        10: 'traffic light',
        11: 'fire hydrant',
        13: 'stop sign',
        14: 'parking meter',
        15: 'bench',
        16: 'bird',
        17: 'cat',
        18: 'dog',
        19: 'horse',
        20: 'sheep',
        21: 'cow',
        22: 'elephant',
        23: 'bear',
        24: 'zebra',
        25: 'giraffe',
        27: 'backpack',
        28: 'umbrella',
        31: 'handbag',
        32: 'tie',
        33: 'suitcase',
        34: 'frisbee',
        35: 'skis',
        36: 'snowboard',
        37: 'sports ball',
        38: 'kite',
        39: 'baseball bat',
        40: 'baseball glove',
        41: 'skateboard',
        42: 'surfboard',
        43: 'tennis racket',
        44: 'bottle',
        46: 'wine glass',
        47: 'cup',
        48: 'fork',
        49: 'knife',
        50: 'spoon',
        51: 'bowl',
        52: 'banana',
        53: 'apple',
        54: 'sandwich',
        55: 'orange',
        56: 'broccoli',
        57: 'carrot',
        58: 'hot dog',
        59: 'pizza',
        60: 'donut',
        61: 'cake',
        62: 'chair',
        63: 'couch',
        64: 'potted plant',
        65: 'bed',
        67: 'dining table',
        70: 'toilet',
        72: 'tv',
        73: 'laptop',
        74: 'mouse',
        75: 'remote',
        76: 'keyboard',
        77: 'cell phone',
        78: 'microwave',
        79: 'oven',
        80: 'toaster',
        81: 'sink',
        82: 'refrigerator',
        84: 'book',
        85: 'clock',
        86: 'vase',
        87: 'scissors',
        88: 'teddy bear',
        89: 'hair drier',
        90: 'toothbrush'
    }
    return mscoco_category2name