# -*- coding: utf-8 -*-
"""DeepFace 本地模型配置，避免重复下载"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEEPFACE_MODELS_PATH = os.path.join(PROJECT_ROOT, ".deepface", "weights")

DEEPFACE_MODEL_MAPPING = {
    "age": "age_model_weights.h5",
    "gender": "gender_model_weights.h5",
    "emotion": "facial_expression_model_weights.h5",
}


def setup_deepface():
    """配置 deepface 使用本地模型"""
    os.environ["DEEPFACE_HOME"] = DEEPFACE_MODELS_PATH
    os.environ["DEEPFACE_WEIGHTS_PATH"] = DEEPFACE_MODELS_PATH
    print(f"  [DeepFace] 模型目录: {DEEPFACE_MODELS_PATH}")
    for name, file in DEEPFACE_MODEL_MAPPING.items():
        path = os.path.join(DEEPFACE_MODELS_PATH, file)
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"    - {name}: {file} {exists}")


def verify_models():
    """验证模型文件完整性"""
    all_ok = True
    for name, file in DEEPFACE_MODEL_MAPPING.items():
        path = os.path.join(DEEPFACE_MODELS_PATH, file)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  ✓ {name}: {file} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {name}: {file} (不存在)")
            all_ok = False
    return all_ok


if __name__ == "__main__":
    verify_models()
