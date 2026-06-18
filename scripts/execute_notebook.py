"""Execute the vectorization notebook using nbclient (no nbconvert/tornado serve)."""
import nbformat
from nbclient import NotebookClient
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
nb_path = PROJECT_ROOT / "notebooks" / "run_vectorization_v008_run1.ipynb"

with open(nb_path, encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(
    nb,
    timeout=300,
    kernel_name="python3",
    resources={"metadata": {"path": str(PROJECT_ROOT)}},
)
client.execute()

with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Notebook executed and saved.")
