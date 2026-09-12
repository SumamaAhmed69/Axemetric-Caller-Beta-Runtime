# Dialforge Cloud Benchmark

This harness benchmarks the local AI path used by Dialforge without installing the Dialforge Windows application on the test computer.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SumamaAhmed69/Axemetric-Caller-Beta-Runtime/blob/main/benchmarks/Dialforge_Cloud_Benchmark.ipynb)

## What it tests

- Ollama + `qwen3:1.7b`
- Ollama + `qwen3:4b`
- Ollama + `qwen3:8b`
- Qwen time to first generated text and tokens/second
- `faster-whisper==1.2.0` with `small.en`
- Chatterbox Nano using the same pinned Chatterbox source revision currently used by the Dialforge runtime publisher
- Synthetic end-to-end turns: prospect audio -> Whisper -> Qwen -> Chatterbox
- GPU memory, GPU utilization, GPU power, RAM and hardware metadata
- Median and P95 latency

The notebook produces:

- `dialforge-benchmark-report.html`
- `dialforge-benchmark-report.json`

The HTML report is shown directly inside Colab. Downloading the reports is optional.

## Free Colab run

1. Open `Dialforge_Cloud_Benchmark.ipynb` with the Colab badge above.
2. Choose **Runtime -> Change runtime type -> GPU**.
3. Choose an available GPU runtime.
4. Use **Runtime -> Run all**.
5. Read the generated performance report at the bottom of the notebook.

The notebook downloads AI models to the temporary Colab VM, not to the local PC. When the Colab runtime is deleted, those model files disappear with it.

## Important scope

This is a compute benchmark for the local AI pipeline. It intentionally does not make phone calls and therefore does not include LiveKit, SIP provider or PSTN network latency. The final Windows beta should still receive a small number of real end-to-end SIP tests before release.

Google controls free Colab GPU availability and may assign different GPU models between sessions. Every report records the exact GPU, VRAM, RAM, CPU, CUDA and driver details so results from different runs can be compared correctly.

## Command-line use

The same runner can be used on any Linux NVIDIA GPU machine after Ollama, the Python dependencies and the pinned Chatterbox source have been installed:

```bash
python benchmarks/dialforge_colab_benchmark.py \
  --models qwen3:1.7b qwen3:4b qwen3:8b \
  --qwen-repeats 3 \
  --component-repeats 3 \
  --pipeline-turns 5 \
  --pipeline-model qwen3:4b
```

Increase `--pipeline-turns` for longer stability runs. The default is intentionally small enough for an ordinary free notebook session while still producing a useful latency profile.
