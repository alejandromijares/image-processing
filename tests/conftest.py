import os
import sys

# Mirror run.sh, which launches from src/ so `from models...` resolves.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
