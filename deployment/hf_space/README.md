---
title: GoEmotions BERT Demo
emoji: 💬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.50.0
python_version: 3.12
app_file: app.py
pinned: false
---

# GoEmotions BERT seed 123

Demo từ checkpoint đã fine-tune; không train và không tải dataset.
Ngưỡng riêng được khóa trên validation, có nút chuyển ngưỡng 0,5.
Lần khởi động đầu tải ZIP checkpoint từ GitHub Release và kiểm SHA-256.
Space chỉ nạp BERT/123, không tự nạp BART hay các kiến trúc khác.
