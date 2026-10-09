"""Ghi ba case đọc lỗi đã đối chiếu, không chạy lại hay sửa dự đoán.

Các nhận xét là quan sát do trợ lý ghi từ văn bản và scores thật.
Thành viên cần tự đọc lại trước khi bảo vệ; không coi đây là nhãn mới.
"""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from update_report_results import ERROR_CATEGORIES, select_error_case_ids

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/errors_test_standard_fixed/examples.csv"
NOTES = {
    "eczj48j": (
        "Văn bản rất ngắn, có dấu chấm than, emoji và cụm 'hard work'. "
        "Đây là các dấu hiệu có thể khiến việc suy ra đủ bộ nhãn khó hơn, "
        "nhưng không chứng minh nguyên nhân trong mô hình. C1 dự đoán caring "
        "và bỏ cả ba nhãn thật; cờ partial=False ở C1 chỉ vì không có TP, "
        "không có nghĩa là dự đoán đúng. C2/C3 nhận ra admiration nhưng bỏ "
        "excitement và neutral. Giữ nguyên ground truth kể cả neutral đồng xuất hiện."
    ),
    "ed0jr9i": (
        "Các từ 'shy', 'awkward' và tình huống đưa danh thiếp là dấu hiệu "
        "ngôn ngữ về sự ngượng ngùng. C1 nhận ra embarrassment; C2/C3 có "
        "score nhãn này dưới 0.5 nên bỏ sót. Embarrassment có 303 mẫu train "
        "và thuộc nhóm năm nhãn hiếm đã xác định từ train. Không suy diễn "
        "cơ chế attention, nguyên nhân do độ dài hoặc tác dụng của weighting "
        "từ riêng một ví dụ."
    ),
    "eczcvgx": (
        "Câu kể về dự định và ý kiến khác của vợ; không có từ thể hiện "
        "sự tán thành rõ ràng. Ground truth là neutral. C1/C2 trả đúng "
        "neutral; C3 chọn approval và bỏ neutral vì hai score nằm ở hai "
        "phía ngưỡng 0.5. Đây là cặp FN neutral / FP approval ở C3, "
        "không phải bằng chứng chắc chắn về mỉa mai hay cảm xúc thật của người viết."
    ),
}
NAMES = {"bert": "C1 BERT", "roberta": "C2 RoBERTa", "distilbert": "C3 DistilBERT"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    before = sha(SOURCE)
    with SOURCE.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    selected = select_error_case_ids(rows)
    if set(selected.values()) != set(NOTES):
        raise ValueError("Bộ case đã đổi; phải đọc lại nội dung trước khi ghi nhận xét")
    changes = []
    for row in rows:
        if selected.get(row["category"]) == row["id"]:
            row["manual_linguistic_notes"] = NOTES[row["id"]]
            changes.append({key: row[key] for key in ("category", "id", "architecture", "seed")})
    if len(changes) != 9:
        raise ValueError("Cần đúng ba ID × ba kiến trúc")
    with SOURCE.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Ba case phân tích lỗi trên test: đối chiếu cùng ID giữa C1/C2/C3",
        "",
        "Dự đoán đã có trước khi đọc lỗi; dùng ngưỡng cố định 0.5. "
        "Checkpoint đại diện của mỗi kiến trúc được chọn bằng validation, "
        "không chọn lại theo các câu test này. Đây là ba ví dụ minh họa "
        "của các nhóm lỗi có thể chồng lấp; không đại diện toàn bộ phân bố lỗi.",
        "",
        "Nhận xét do trợ lý ghi sau khi đọc văn bản, nhãn thật và 28 scores đã lưu. "
        "Nhóm cần tự xác nhận cách diễn giải trước khi nộp/bảo vệ. "
        "Không thay nhãn thật, scores, ngưỡng hay cấu hình sau khi xem test.",
    ]
    for index, category in enumerate(ERROR_CATEGORIES, 1):
        case_id = selected[category]
        matched = [row for row in rows if row["category"] == category and row["id"] == case_id]
        matched.sort(key=lambda row: ("bert", "roberta", "distilbert").index(row["architecture"]))
        lines += ["", f"## Case {index}: {category} — ID {case_id}", "",
                  f"Văn bản: “{matched[0]['text']}”.", "",
                  "Nhãn thật: " + ", ".join(json.loads(matched[0]["true_labels"])) + ".", "",
                  f"**Bảng 5-2{chr(ord('g') + index - 1)}. Ba C trên cùng ID {case_id}.**", "",
                  "| C | Seed | Dự đoán | Bỏ sót (FN) | Nhãn thừa (FP) | Scores liên quan | Cờ nhóm lỗi |",
                  "|---|---:|---|---|---|---|---|"]
        for row in matched:
            predicted, missed, extra = [json.loads(row[key]) for key in ("predicted_labels", "missed_labels", "extra_labels")]
            scores = json.loads(row["scores_by_label"])
            related = list(dict.fromkeys(json.loads(row["true_labels"]) + extra))
            values = [NAMES[row["architecture"]], row["seed"], ", ".join(predicted) or "không nhãn",
                      ", ".join(missed) or "không", ", ".join(extra) or "không",
                      "; ".join(f"{label}={scores[label]:.4f}" for label in related), row["error_present"]]
            lines.append("| " + " | ".join(str(value).replace("|", "/") for value in values) + " |")
        lines += ["", "Nhận xét: " + NOTES[case_id]]
    lines += ["", "## Cách giải thích khi bảo vệ", "",
              "Cờ nhóm lỗi chỉ trả lời mẫu có thuộc đúng định nghĩa nhóm đó hay không; "
              "cờ False không chứng minh mọi nhãn đều đúng. Ví dụ case 1, C1 sai hoàn toàn "
              "nhưng không thuộc lỗi nhận được một phần nhãn. C1 thắng trung bình theo "
              "tiêu chí chọn trên validation vẫn có thể thua ở một câu riêng.", "",
              "Scores là đầu ra sigmoid của các classifier, không phải xác suất đã "
              "được kiểm chuẩn. Các quan sát trên không xác lập quan hệ nhân quả. "
              "Số lỗi toàn tập xem group_summary.csv; không cộng các nhóm chồng lấp.", "",
              "Nguồn đối chiếu: reports/errors_test_standard_fixed/examples.csv và manifest.json. "
              "Scores trong bảng làm tròn bốn chữ số; CSV giữ độ chính xác gốc.", ""]
    output = ROOT / "reports/error_case_studies.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    evidence = {"annotated_at_utc": datetime.now(timezone.utc).isoformat(),
                "reviewer": "Codex assistant; team confirmation pending",
                "source": SOURCE.relative_to(ROOT).as_posix(), "source_sha256_before": before,
                "source_sha256_after": sha(SOURCE), "updated_rows": changes,
                "columns_changed": ["manual_linguistic_notes"], "fit_or_inference_executed": False,
                "case_studies": output.relative_to(ROOT).as_posix(), "case_studies_sha256": sha(output)}
    (ROOT / "reports/execution/error_annotation.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Đã ghi nhận xét cho 3 ID × 3 C; giữ nguyên dự đoán.")


if __name__ == "__main__":
    main()
