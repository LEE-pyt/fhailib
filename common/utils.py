import os
import yaml


def get_meta_data(data_folder, project_name) -> dict:
    project_folder = os.path.join(data_folder, project_name)
    meta_folder = os.path.join(project_folder, "meta")
    meta_file = os.path.join(meta_folder, "meta.yml")

    # YAML 파일을 열고 내용을 파싱하여 Python 사전으로 반환
    with open(meta_file, "r", encoding="utf-8") as meta:
        meta_dict = yaml.safe_load(meta)  # YAML 내용을 안전하게 파싱

    return meta_dict


def write_meta_data(data_folder, project_name, key, value) -> dict:
    project_folder = os.path.join(data_folder, project_name)
    meta_folder = os.path.join(project_folder, "meta")
    os.makedirs(meta_folder, exist_ok=True)  # 메타 폴더가 없으면 생성
    meta_file = os.path.join(meta_folder, "meta.yml")

    # 메타 파일이 이미 존재하면, 기존 내용을 로드. 그렇지 않으면 빈 사전 생성
    if os.path.exists(meta_file):
        with open(meta_file, "r", encoding="utf-8") as meta:
            meta_dict = yaml.safe_load(meta) or {}
    else:
        meta_dict = {}

    # 새로운 키와 값을 사전에 추가
    meta_dict[key] = value

    # 변경된 사전을 YAML 형식으로 파일에 다시 쓰기
    with open(meta_file, "w", encoding="utf-8") as meta:
        yaml.dump(meta_dict, meta, allow_unicode=True)

    return meta_dict
