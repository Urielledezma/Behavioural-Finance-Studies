import pathlib
import sys

# The study's library is imported from its src/ rather than installed, as the notebook does.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
