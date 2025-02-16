# src/config.py
import os

INPUT_FILE_PATH = os.getenv('INPUT_FILE_PATH', './data/input/test_1.json')
OUTPUT_FILE_PATH = os.getenv('OUTPUT_FILE_PATH', './data/output/output.json')
DEFAULT_K = int(os.getenv('DEFAULT_K', 4))
