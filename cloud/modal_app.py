"""Optional Modal launcher. Review the generated job and secrets before use."""
from pathlib import Path

try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App("ioo-manifold-fresh")
    image = modal.Image.debian_slim(python_version="3.11").pip_install(
        "numpy", "scipy", "scikit-learn", "matplotlib")

    @app.function(image=image, timeout=900)
    def run_seed(seed: int, output: str = "/tmp/ioo"):
        import sys
        sys.path.insert(0, "/root/ioo")
        from src.ioo_experiment import run
        return run(output, [seed], machines=16, episodes=8, steps=800)

    @app.local_entrypoint()
    def main(seeds: int = 5):
        for result in run_seed.map(range(seeds)):
            print(result)
else:
    app = None

if __name__ == "__main__" and modal is None:
    raise SystemExit("Install modal to use this optional launcher: pip install modal")
