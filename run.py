"""Điểm vào của chương trình. Ví dụ:

    py run.py                              # demo mặc định (config/snack.yaml)
    py run.py --config config/coconut.yaml
    py run.py --no-display --save-video    # chạy nền và xuất video kết quả
"""
import argparse
import sys
from pathlib import Path

from core.logging_setup import setup_logging
from core.settings_loader import ConfigError, load_settings


def parse_args():
    parser = argparse.ArgumentParser(description="Đếm vật trên băng chuyền bằng YOLO và ByteTrack")
    parser.add_argument("--config", help="File preset, mặc định config/snack.yaml")
    parser.add_argument("--input", help="Đường dẫn video, thay cho input.path")
    parser.add_argument("--model", help="Tên model trong models/ (không có .pt), thay cho detector.type")
    parser.add_argument("--conf", type=float, help="Ngưỡng confidence, thay cho detector.conf_threshold")
    parser.add_argument("--stride", type=int, help="Xử lý mỗi N frame, thay cho input.vid_stride")
    parser.add_argument("--no-display", action="store_true", help="Không mở cửa sổ hiển thị")
    parser.add_argument("--save-video", action="store_true", help="Lưu video kết quả vào outputs/")
    return parser.parse_args()


def from_cwd(path):
    """Đường dẫn gõ trên dòng lệnh được hiểu theo thư mục hiện tại nếu file đó tồn tại ở đây."""
    candidate = Path(path).resolve()
    return str(candidate) if candidate.exists() else path


def build_overrides(args):
    overrides = {}
    if args.input:
        overrides[("input", "path")] = from_cwd(args.input)
    if args.model:
        overrides[("detector", "type")] = args.model
    if args.conf is not None:
        overrides[("detector", "conf_threshold")] = args.conf
    if args.stride is not None:
        overrides[("input", "vid_stride")] = args.stride
    if args.no_display:
        overrides[("input", "display")] = False
    if args.save_video:
        overrides[("saver", "save_videos")] = True
    return overrides


def main():
    # Console Windows có thể không dùng UTF-8; tránh lỗi khi in tiếng Việt.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):  # không có khi stdout bị thay (IDE, pytest, pipe đặc biệt)
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()

    try:
        config = load_settings(from_cwd(args.config) if args.config else None, build_overrides(args))
    except ConfigError as error:
        print(error, file=sys.stderr)
        return 1

    setup_logging()

    # Import muộn: ultralytics tải khá lâu, không cần chờ nếu cấu hình sai hoặc chỉ gọi --help.
    from src.pipeline import DetectionPipeline

    pipeline = DetectionPipeline(config)
    summary = pipeline.run_video(
        config["input"]["path"],
        vid_stride=config["input"].get("vid_stride", 1),
        display=config["input"].get("display", True),
        realtime=config["input"].get("realtime", True),
    )

    for item in summary["counts"]:
        print(f"Kết quả [{item['type']}]: {item['count']}")
    if summary["output_dir"]:
        print(f"Đã lưu kết quả tại: {summary['output_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
