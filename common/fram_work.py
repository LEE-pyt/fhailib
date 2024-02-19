import os
import yaml  # PyYAML 라이브러리를 임포트
from utils import *
from enum import Enum
import os


class Task(Enum):
    OBJECT_DETECTION = "object_detection"
    ANOMALY_DETECTION = "anomaly_detection"


def init_project(data_folder, project_name, task):
    # 태스크가 유효한지 확인
    if task not in Task.__members__.values():
        print("Invalid task specified.")
        return False

    project_folder = os.path.join(data_folder, project_name)
    img_folder = os.path.join(project_folder, "images")
    label_folder = os.path.join(project_folder, "labels")
    meta_folder = os.path.join(project_folder, "meta")
    os.makedirs(img_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)
    os.makedirs(meta_folder, exist_ok=True)

    if task == Task.anomaly_detection:
        good_folder = os.path.join(img_folder, "good")
        bad_folder = os.path.join(img_folder, "bad")
        os.makedirs(good_folder, exist_ok=True)
        os.makedirs(bad_folder, exist_ok=True)

    meta_file = os.path.join(meta_folder, "meta.yml")
    # 메타 파일에 YAML 형식으로 태스크 정보 작성
    meta_content = {"task": task, "project_name": project_name}

    with open(meta_file, "w", encoding="utf-8") as meta:
        # YAML 형식으로 변환하여 파일에 쓰기
        meta.write(f"task: {meta_content['task']}\n")
        meta.write(f"description: {meta_content['project_name']}\n")
