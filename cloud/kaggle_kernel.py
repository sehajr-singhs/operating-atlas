"""Generate a self-contained Kaggle kernel for the fresh IOO benchmark."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "kaggle_submission"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = (ROOT / "src" / "ioo_experiment.py").read_text(encoding="utf-8")
    core = (ROOT / "src" / "ioo_core.py").read_text(encoding="utf-8")
    main_code = f'''# Generated fresh IOO Kaggle kernel\nimport sys, json\nfrom pathlib import Path\nPath("ioo_kernel_core.py").write_text({core!r})\nPath("ioo_kernel_experiment.py").write_text({source!r})\nsys.path.insert(0, ".")\nimport ioo_kernel_experiment as experiment\nresult = experiment.run("/kaggle/working/ioo_results", seeds=[0,1,2,3,4], machines=16, episodes=8, steps=800)\nprint(json.dumps(result, indent=2))\n'''
    (OUT / "main.py").write_text(main_code, encoding="utf-8")
    metadata = {
        "id": "YOUR_USERNAME/ioo-manifold-fresh",
        "title": "IOO Manifold Fresh Benchmark",
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    (OUT / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
