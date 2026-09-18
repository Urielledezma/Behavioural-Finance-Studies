import pathlib
import sys

# The library is imported from src/ rather than installed, as the notebooks do.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
