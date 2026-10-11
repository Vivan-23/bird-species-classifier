import json
import re
from pathlib import Path
import nbformat
from nbformat.validator import normalize
from nbclient import NotebookClient

PROJECT_ROOT = Path.cwd()
notebook_path = PROJECT_ROOT / "Bird_Species_Classification_Improved.ipynb"

print(f"[1] Loading notebook from {notebook_path}...")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# 1. Clean up corrupted dash characters in all cells
print("[2] Fixing character encoding / mojibake in text and titles...")
for cell in nb.cells:
    # Replace any non-standard or corrupted dash sequences
    cell.source = cell.source.replace("\u2014", " - ")
    cell.source = cell.source.replace("\u2013", "-")
    cell.source = cell.source.replace("", " - ")

# 2. Fix Cell 26 (Index 25) to define class_to_idx
print("[3] Fixing Cell 26 missing class_to_idx definition...")
c26 = nb.cells[25]
c26_source = c26.source

if "class_to_idx = " not in c26_source:
    # Insert class_to_idx definition before train_dataset_transfer
    insert_point = "train_dataset_transfer = BirdDataset("
    replacement = "class_to_idx = {name: idx for idx, name in enumerate(classes)}\n\n" + insert_point
    c26.source = c26_source.replace(insert_point, replacement)
    print("    Inserted class_to_idx definition into Cell 26.")

# Also adjust MAX_EPOCHS to 6 for CPU feasibility
c27 = nb.cells[26]
c27.source = c27.source.replace("MAX_EPOCHS = 30", "MAX_EPOCHS = 6")
c27.source = c27.source.replace("PATIENCE = 7", "PATIENCE = 3")

# In Cell 26 also ensure MAX_EPOCHS is 6
c26.source = c26.source.replace("MAX_EPOCHS = 30", "MAX_EPOCHS = 6")
c26.source = c26.source.replace("PATIENCE = 7", "PATIENCE = 3")

# 3. Clean up titles and headings in Markdown cells
nb.cells[0].source = nb.cells[0].source.replace("# COURSERA  -  TT2 Assignment", "# COURSERA - TT2 Assignment")
nb.cells[21].source = nb.cells[21].source.replace("Section K  -  Conclusion", "Section K - Conclusion")
nb.cells[22].source = nb.cells[22].source.replace("Section L  -  References", "Section L - References")
nb.cells[23].source = nb.cells[23].source.replace("Section L  -  Data Integrity", "Section M - Data Integrity")
nb.cells[27].source = nb.cells[27].source.replace("Section O  -  One-Time", "Section O - One-Time")
nb.cells[29].source = nb.cells[29].source.replace("Section P  -  Results", "Section P - Results")

# Normalize notebook structure to satisfy all nbformat schema requirements
normalize(nb)

# Save the updated notebook
with open(notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"[4] Successfully updated and normalized {notebook_path}!")
