from pathlib import Path

from ultralytics import YOLO


HERE = Path(__file__).resolve().parent
SOURCE_MODEL = HERE / "best.pt"


def main() -> None:
    if not SOURCE_MODEL.is_file():
        raise FileNotFoundError(f"Model not found: {SOURCE_MODEL}")

    model = YOLO(str(SOURCE_MODEL))

    exported_path = model.export(
        format="openvino",
        imgsz=960,
        dynamic=False,
        half=False,
        batch=1,
    )

    print("OpenVINO export completed.")
    print("Exported model:", exported_path)


if __name__ == "__main__":
    main()