from helpers import read_config
config = read_config("config.ini")

# Test reading a value from each section
parent_dir = config.get('paths', 'parent_dir')
print(f"parent_dir: {parent_dir}")

selection_pct = config.getfloat('parameters', 'selection_pct')
print(f"selection_pct: {selection_pct}")
