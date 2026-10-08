"""Chèn bảng số từ artifacts thật vào báo cáo; không tính lại/chọn bằng test."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    summary = json.loads((ROOT / "reports/project_results/summary.json").read_text(encoding="utf-8"))
    content = (ROOT / "reports/project_results/RESULTS.md").read_text(encoding="utf-8")
    content = content.replace("# Kết quả thí nghiệm thực tế", "**Bảng 5-2. Kết quả A/B/C thực tế.**", 1)
    content = content.replace("## Phần chưa có bằng chứng đầy đủ", "### 5.2.1. Kiểm tra mức hoàn thành")
    additional = ["", "### 5.2.2. Cấu hình, seed và lựa chọn demo", ""]
    selection_path = ROOT / "data/processed/transformers/selected_model.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        additional.append("Danh tính mô hình được chọn lưu trong selected_model.json; lựa chọn dựa trên mean validation Macro-F1@0,5 của ba seed, không dựa trên test.")
        additional.append("```json\n" + json.dumps(selection, ensure_ascii=False, indent=2) + "\n```")
    else:
        additional.append("Chưa đủ hồ sơ để chọn demo C; không dùng kết quả smoke hoặc A/B để thay thế.")
    extra_paths = [("reports/errors_test_standard_fixed/summary.md", "### 5.2.3. Ba nhóm lỗi C1/C2/C3 trên test"),
                   ("reports/project_results/ANALYSIS.md", "### 5.2.4. Nhãn hiếm và chi phí huấn luyện")]
    for filename, title in extra_paths:
        path = ROOT / filename
        if path.exists():
            extra = path.read_text(encoding="utf-8")
            extra = re.sub(r"^# .*\n", "", extra, count=1)
            extra = re.sub(r"^## ", "#### ", extra, flags=re.MULTILINE)
            def portable_image(match):
                image_path = Path(match[2])
                try:
                    target = image_path.relative_to(ROOT / "reports").as_posix()
                except ValueError:
                    target = image_path.as_posix()
                return f"![{match[1]}]({target})"
            extra = re.sub(r"!\[([^\]]*)\]\(<([^>]+)>\)", portable_image, extra)
            additional.extend(["", title, "", extra])
    evidence = ROOT / "reports/demo_verification.json"
    if evidence.exists():
        verified = json.loads(evidence.read_text(encoding="utf-8"))
        additional.extend(["", "### 5.2.5. Kiểm giao diện demo", "", "```json",
                           json.dumps(verified, ensure_ascii=False, indent=2), "```"])
    replacement = "<!-- AUTO_RESULTS -->\n" + content + "\n" + "\n".join(additional) + "\n<!-- END_AUTO_RESULTS -->"
    source = ROOT / "reports/BAO_CAO_DO_AN_NOI_DUNG.md"
    report = source.read_text(encoding="utf-8")
    report, count = re.subn(r"<!-- AUTO_RESULTS -->.*?<!-- END_AUTO_RESULTS -->", lambda _: replacement,
                           report, flags=re.DOTALL)
    if count != 1:
        raise ValueError("Cần đúng một khối AUTO_RESULTS")
    if summary["complete"]:
        report = report.replace("Các kết quả B/C/D chỉ được bổ sung từ tệp chạy thật; không suy ra kết quả thực nghiệm từ việc có mã nguồn.",
                                "Toàn bộ A/B/C đã có kết quả full; mỗi kiến trúc C gồm ba seed 42,123,2026 với mean và sample std. Bảng test sử dụng ngưỡng/mô hình đã khóa trên validation.")
        report = report.replace("Kết quả A hiện dùng validation; tuning trên cùng validation có thể lạc quan.",
                                "Bảng test đã chạy sau khóa protocol; bảng tuned-validation dùng lại dữ liệu chọn ngưỡng nên có thể lạc quan.")
        report = report.replace("Ưu tiên hoàn tất đủ B/C, seed, lỗi đối chiếu và demo; khóa model/ngưỡng rồi đánh giá test theo protocol.",
                                "Duy trì hồ sơ A/B/C đủ seed, kiểm demo khi chuyển máy và đọc thủ công ví dụ lỗi; không chọn lại model/ngưỡng bằng test.")
        paragraph = "Các thí nghiệm A/B/C đã chạy full và khóa protocol trước test. Ba kiến trúc C được huấn luyện bằng cùng split và ba seed; bảng5-2 lưu từng cấu hình cùng mean±std. Phân tích lỗi đối chiếu checkpoint đại diện chọn trên validation. Kết quả demo cần được kiểm trực tiếp bằng hồ sơ đi kèm; bảng phân công và tỷ lệ đóng góp vẫn cần nhóm xác nhận."
        report = re.sub(r"Thiết kế toàn đồ án bao gồm B, ba C và D theo phân công bốn người\..*?(?=\n\n)", paragraph, report, count=1, flags=re.DOTALL)
    source.write_text(report, encoding="utf-8")
    print("Cập nhật báo cáo từ số thực nghiệm; complete =", summary["complete"])


if __name__ == "__main__":
    main()
