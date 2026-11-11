"""Put examples/ on sys.path so the equivalence tests can import the analysis drivers by name."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"))
