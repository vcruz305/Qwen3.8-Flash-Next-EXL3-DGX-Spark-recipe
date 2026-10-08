# Matrix runner publication files

Copy `bench/run_matrix.py`, `bench/test_run_matrix.py` and `bench/matrix.md` into the corresponding recipe paths. Add a link to `matrix.md` from the benchmark README when integrating. This staging directory does not modify the root checkout.

The runner is byte-identical to the frozen overnight source: SHA256 `48281d5b51b1b64509385548c6f41f34909c328070274fcb4a06945ddcc44586`. The existing 20 CPU tests change only the adjacent imported filename. They pass both here and under isolated Python after copying into an unrelated temporary directory; `--help` works there too. No GPU, API, service or Spark action occurs during packaging.

`packaging-validation.json` and `cpu-validation.log` retain the offline verification. The usage document describes the exact existing resume/cleanup contract, actual source versus tracking-label identity, and the original 6fbc0a2 launcher incompatibility without altering controller behavior. Historical original API controllers are archived separately; they do not belong in the normal executable runner path.
