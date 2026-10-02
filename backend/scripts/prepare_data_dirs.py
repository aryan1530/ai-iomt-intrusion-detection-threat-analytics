"""Place public datasets under data/raw/{nsl_kdd,ciciomt2024,wustl_ehms}.

Expected:
  NSL-KDD: KDDTrain+.txt, KDDTest+.txt
    https://www.unb.ca/cic/datasets/nsl.html
  CICIoMT2024: CSV flows from Canadian Institute for Cybersecurity
    https://www.unb.ca/cic/datasets/iomt-dataset-2024.html
  WUSTL-EHMS-2020: CSV from WUSTL
    https://www.cse.wustl.edu/~jain/ehms/index.html

If files are missing, loaders generate representative synthetic traffic for demo training.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for name in ["nsl_kdd", "ciciomt2024", "wustl_ehms"]:
    (ROOT / "data" / "raw" / name).mkdir(parents=True, exist_ok=True)
    readme = ROOT / "data" / "raw" / name / "README.txt"
    if not readme.exists():
        readme.write_text(f"Drop official CSV/TXT files for {name} here.\n")
print("Dataset folders ready at", ROOT / "data" / "raw")
