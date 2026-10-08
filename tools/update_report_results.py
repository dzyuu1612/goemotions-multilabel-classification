"""Chèn bảng số từ artifacts thật vào báo cáo; không tính lại/chọn bằng test."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def final_abstract(summary):
    """Tóm tắt chỉ có số test khi bảng thực nghiệm đã qua đủ các điều kiện."""
    if not summary["complete"]:
        return None
    lookup = {(r["system"], r["split"], r["threshold_mode"]): r
              for r in summary["averages"]}
    selection = json.loads((ROOT / "data/processed/transformers/selected_model.json").read_text(encoding="utf-8"))
    system = "C_" + selection["architecture"]
    before = lookup[("A_standard", "test", "fixed")]
    after = lookup[("A_balanced", "test", "tuned")]
    selected = lookup[(system, "test", "fixed")]
    tuned = lookup[(system, "test", "tuned")]
    if selected["n_runs"] != 3 or tuned["n_runs"] != 3:
        raise ValueError("Tóm tắt C cần đủ ba seed; không thay bằng điểm checkpoint demo")
    return (f"Phần A đã được đo trên toàn bộ 5.427 mẫu test sau khi khóa cấu hình trên validation. "
            f"Bản standard với ngưỡng 0,5 đạt Macro-F1 {before['macro_f1_mean']:.4f}; "
            f"bản balanced với ngưỡng riêng đạt {after['macro_f1_mean']:.4f}. "
            f"Kiến trúc C được chọn bằng mean Macro-F1 validation @0,5 là {selection['architecture']}; "
            f"trên test, kiến trúc này đạt Macro-F1 {selected['macro_f1_mean']:.4f} ± {selected['macro_f1_std']:.4f} "
            f"và Micro-F1 {selected['micro_f1_mean']:.4f} ± {selected['micro_f1_std']:.4f}. "
            f"Ngưỡng riêng chọn trên validation đưa Macro-F1 test tới "
            f"{tuned['macro_f1_mean']:.4f} ± {tuned['macro_f1_std']:.4f} "
            f"(chênh lệch mean {tuned['macro_f1_mean'] - selected['macro_f1_mean']:+.4f}). "
            f"Mean và sample std tính giữa ba seed 42, 123, 2026; seed {selection['seed']} "
            "của demo là một checkpoint đại diện, không phải ensemble hay điểm trung bình. "
            "Báo cáo giữ cả các thay đổi F1 âm của nhãn hiếm, đối chiếu lỗi giữa ba C và thảo luận giá trị "
            "ứng dụng dự kiến. Kết quả này chưa xác nhận hiệu quả trên dữ liệu tiếng Việt hoặc ROI công nghiệp.")


def interpret_results(summary):
    """Nhận xét chỉ từ bảng thật; thứ hạng C dùng validation đã quy định."""
    rows = summary["averages"]
    original = [r for r in rows if r["system"].startswith("C_")
                and r["split"] == "validation" and r["threshold_mode"] == "fixed"]
    if len(original) != 3 or any(r["n_runs"] != 3 for r in original):
        return []
    ordered = sorted(original, key=lambda r: -r["macro_f1_mean"])
    best, worst = ordered[0], ordered[-1]
    stable = min(original, key=lambda r: r["macro_f1_std"])
    lines = ["", "### 5.2.6. Thứ hạng, độ ổn định và đánh đổi", "",
             f"Theo tiêu chí mean Macro-F1 validation @0,5 đã chốt, {best['system']} "
             f"đạt cao nhất ({best['macro_f1_mean']:.4f} ± {best['macro_f1_std']:.4f}); "
             f"{worst['system']} thấp nhất ({worst['macro_f1_mean']:.4f} ± {worst['macro_f1_std']:.4f}). "
             "Đây là thứ hạng trong ba cấu hình đã thử, không phải khẳng định một kiến trúc luôn tốt nhất.",
             f"{stable['system']} có sample std Macro-F1 nhỏ nhất trên validation "
             f"({stable['macro_f1_std']:.4f}). Std được tính giữa ba seed, khác biến động giữa các nhãn. "
             "Ba seed giúp mô tả độ ổn định trong lần đo nhưng chưa đủ để suy ra ý nghĩa thống kê hoặc bảo đảm tái hiện trên mọi máy."]
    for system in (r["system"] for r in original):
        test = {r["threshold_mode"]: r for r in rows
                if r["system"] == system and r["split"] == "test"}
        if "fixed" in test and "tuned" in test:
            before, after = test["fixed"], test["tuned"]
            change = after["macro_f1_mean"] - before["macro_f1_mean"]
            lines.append(f"{system} trên test: Macro-F1 @0,5 "
                         f"{before['macro_f1_mean']:.4f} ± {before['macro_f1_std']:.4f}; "
                         f"ngưỡng riêng {after['macro_f1_mean']:.4f} ± {after['macro_f1_std']:.4f} "
                         f"(chênh lệch mean {change:+.4f}). Micro-F1 tương ứng "
                         f"{before['micro_f1_mean']:.4f} và {after['micro_f1_mean']:.4f}. "
                         "Ngưỡng được chọn trên validation của từng seed, không chọn lại trên test.")
    lines.append("Nguyên nhân thứ hạng cần xét cùng dữ liệu, tokenizer, số tham số, learning rate, số epoch và ví dụ lỗi; "
                 "các kết quả này chưa tách riêng ảnh hưởng của từng yếu tố. Xem P/R và Hamming cùng F1: "
                 "hạ ngưỡng có thể tăng recall nhưng thêm false positives. Nhãn hiếm có support nhỏ nên F1 dễ thay đổi; "
                 "giữ cả nhãn giảm điểm trong bảng trước/sau. Thời gian BERT seed 42 có gián đoạn máy ngủ, "
                 "vì vậy không dùng bảng elapsed để xếp hạng tốc độ các kiến trúc.")
    return lines


def seed_table(summary):
    """Giữ từng seed bên cạnh bảng mean/std, kể cả khi mới có một run full."""
    records = summary["records"]
    originals = [r for r in records if r["system"].startswith("C_")
                 and r["split"] == "validation" and r["threshold_mode"] == "fixed"]
    if not originals:
        return []
    lookup = {(r["system"], r["seed"], r["split"], r["threshold_mode"]): r for r in records}
    lines = ["", "**Bảng 5-2f. Kết quả từng seed C đã hoàn tất; Macro-F1.**", "",
             "| Kiến trúc | Seed | Epoch chọn | Val @0,5 | Test @0,5 | Test ngưỡng riêng |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in originals:
        architecture = row["system"].removeprefix("C_")
        folder = ROOT / "data/processed/transformers" / architecture / f"seed_{row['seed']}" / "full/standard"
        metadata = json.loads((folder / "run_metadata.json").read_text(encoding="utf-8"))
        def test_value(mode):
            value = lookup.get((row["system"], row["seed"], "test", mode))
            return f"{value['macro_f1']:.4f}" if value else "Chưa đo"
        lines.append(f"| {architecture} | {row['seed']} | {metadata['selected_epoch']} | "
                     f"{row['macro_f1']:.4f} | {test_value('fixed')} | {test_value('tuned')} |")
    lines.extend(["", "Epoch chọn theo validation @0,5 của đúng seed. File all_runs.csv giữ đủ bảy metrics "
                  "cho từng seed, split và luật ngưỡng; mean_std.csv giữ sample std. Những run chưa hoàn tất "
                  "không được tính vào bảng. Cấu hình/revision/hash nhỏ lưu trong reports/reproducibility; "
                  "trọng số lớn nằm trong data/processed để chạy demo hoặc chia sẻ riêng."])
    return lines


def main():
    summary = json.loads((ROOT / "reports/project_results/summary.json").read_text(encoding="utf-8"))
    content = (ROOT / "reports/project_results/RESULTS.md").read_text(encoding="utf-8")
    content = content.replace("# Kết quả thí nghiệm thực tế", "**Bảng 5-2a. Kết quả A/B/C thực tế.**", 1)
    content = content.replace("Bảng 5-3. Precision/Recall", "Bảng 5-2b. Precision/Recall")
    content = content.replace("## Phần chưa có bằng chứng đầy đủ", "### 5.2.1. Kiểm tra mức hoàn thành")
    additional = ["", "### 5.2.2. Cấu hình, seed và lựa chọn demo", ""]
    selection_path = ROOT / "data/processed/transformers/selected_model.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        additional.append("Danh tính mô hình được chọn lưu trong selected_model.json; lựa chọn dựa trên mean validation Macro-F1@0,5 của ba seed, không dựa trên test.")
        selected_summary = {key: selection[key] for key in
                            ("architecture", "seed", "run_dir", "selection_rule", "checkpoint_rule", "default_threshold")}
        additional.append("```json\n" + json.dumps(selected_summary, ensure_ascii=False, indent=2) + "\n```")
    else:
        additional.append("Chưa đủ hồ sơ để chọn demo C; không dùng kết quả smoke hoặc A/B để thay thế.")
    additional.extend(seed_table(summary))
    extra_paths = [("reports/errors_test_standard_fixed/summary.md", "### 5.2.3. Ba nhóm lỗi C1/C2/C3 trên test"),
                   ("reports/error_case_studies.md", "#### 5.2.3.1. Đọc và giải thích các ví dụ lỗi cụ thể"),
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
            if "errors_test" in filename:
                extra = extra.replace("| Kiến trúc | Seed | Nhóm lỗi", "**Bảng 5-2c. So sánh ba nhóm lỗi trên test.**\n\n| Kiến trúc | Seed | Nhóm lỗi", 1)
            else:
                extra = extra.replace("| Nhãn | Mô hình |", "**Bảng 5-2d. F1 năm nhãn hiếm trước/sau cải tiến.**\n\n| Nhãn | Mô hình |", 1)
                extra = extra.replace("| Kiến trúc | Full seeds", "**Bảng 5-2e. Thời gian hoàn thành run C và số tham số.**\n\n| Kiến trúc | Full seeds", 1)
                extra = re.sub(r"(!\[[^\]]+\]\([^\n]+\))", r"\1\n\n**Hình 5.1. Đường học validation: mean và sample std theo epoch.**", extra, count=1)
            additional.extend(["", title, "", extra])
    evidence = ROOT / "reports/demo_verification.json"
    if evidence.exists():
        verified = json.loads(evidence.read_text(encoding="utf-8"))
        additional.extend(["", "### 5.2.5. Kiểm giao diện demo", "", "```json",
                           json.dumps(verified, ensure_ascii=False, indent=2), "```"])
    additional.extend(interpret_results(summary))
    replacement = "<!-- AUTO_RESULTS -->\n" + content + "\n" + "\n".join(additional) + "\n<!-- END_AUTO_RESULTS -->"
    source = ROOT / "reports/BAO_CAO_DO_AN_NOI_DUNG.md"
    report = source.read_text(encoding="utf-8")
    report, count = re.subn(r"<!-- AUTO_RESULTS -->.*?<!-- END_AUTO_RESULTS -->", lambda _: replacement,
                           report, flags=re.DOTALL)
    if count != 1:
        raise ValueError("Cần đúng một khối AUTO_RESULTS")
    if summary["complete"]:
        report, abstract_count = re.subn(r"Phần A đã được đo trên toàn bộ .*?(?=\n\n)",
                                        lambda _: final_abstract(summary), report, count=1, flags=re.DOTALL)
        if abstract_count != 1:
            raise ValueError("Không tìm thấy đoạn số liệu trong tóm tắt báo cáo")
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
