"""Smoke-test the parser on a single local sample."""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).parent))
import parse_ebb_results as p

sample = Path(r"C:\Users\kahad\IdeaProjects\EuroBigBrain\runs\harvested_test\ebb_w1_SAMPLE_RC3_20260421_231259.htm")
rec = p.parse_htm(sample)
print(json.dumps(rec, indent=2, default=str))
