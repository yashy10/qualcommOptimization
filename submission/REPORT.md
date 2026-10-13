# Benson Edge final submission

Selected run: `b3_kd_lr001_e3_int8`.

The current technical report is [technical_report.pdf](technical_report.pdf)
(two-page paper plus dashboard appendix); [technical_report_2page.pdf](technical_report_2page.pdf)
is the two-page competition version. Both describe the selected depth-pruned,
three-epoch recovered INT8 model. The full provenance is in [metrics.json](metrics.json).

- Accuracy: 98.84% on all 27,000 EuroSAT images.
- Worst class: Pasture, 97.70%.
- Size: 2.052 MiB; 1.8338 million ONNX parameters; 0.2627 billion MACs.
- Local median latency: 1.110 ms, provisional under background CPU load.
- Prior Q3a and one-epoch B3 results remain historical comparisons.

Reproduction: place supplied student/teacher checkpoints and EuroSAT data in `code/`,
install its requirements, and run `sh reproduce_selected.sh` there in a fresh workspace.
The original three-epoch run trained on MPS. Do not treat CPU or cross-device retraining
as guaranteed byte-identical. The provided final ONNX hash is recorded in metrics.json.
