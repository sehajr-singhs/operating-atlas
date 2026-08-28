# IOO Google Colab run

```python
from google.colab import files
# Upload the ioo-manifold folder as a zip, then extract it.
!unzip -q ioo-manifold.zip -d /content/
%cd /content/ioo-manifold
!python -m src.run_experiment --output results/colab --seeds 0 1 2 3 4
```

The run is CPU-compatible. If a GPU is enabled, it can be used by a future neural GNN extension, but the controlled feature benchmark intentionally has no hidden GPU dependency. Download artifacts:

```python
from google.colab import files
files.download('results/colab/metrics.json')
```

Do not place Kaggle or Hugging Face tokens in notebook source. Use Colab Secrets and pass them to an explicitly reviewed connector only.
