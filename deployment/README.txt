GOEMOTIONS WEB DEMO

1. Extract the whole GoEmotions_Web.zip to a writable folder.
2. Install Python 3.12 or 3.13 for Windows if absent (with the py launcher).
3. Double click START_DEMO.cmd. First setup needs Internet for dependencies.
4. Open http://127.0.0.1:7870. Enter a NEW English sentence, click prediction.
5. Close the command window or press Ctrl+C to stop.

The .goemotions file already contains trained BERT seed 123, its original
tokenizer, 28 label names, and per-label validation thresholds. It is a ZIP
container, not a CSV, installer, or a new training run. Do not edit its contents.
It expands to a verified local cache. Inference does not need Internet after
Python dependencies are installed. No Reddit dataset or notebooks are needed.

This package does not include Python itself. It is not a standalone MSI/EXE.
CPU float32 inference can differ slightly from original GPU AMP scores.
The UI can export 28 prediction scores to CSV. Those scores are not model weights.
The five example sentences are authored demonstration prompts, not test labels.
Project source and original checkpoint provenance are in the repository README.
