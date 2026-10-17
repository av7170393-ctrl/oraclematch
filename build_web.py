"""Assemble ../index.html = template.html + data.json (run build_data.py first, or this script runs it)."""
import json
from pathlib import Path
import build_data

HERE = Path(__file__).resolve().parent
data_file = HERE / "data.json"
data = json.loads(data_file.read_text()) if data_file.exists() else build_data.build()
payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
html = (HERE / "template.html").read_text().replace("const D=__DATA__;", "const D=" + payload + ";")
(HERE.parent / "index.html").write_text(html)
print(f"index.html written: {len(html)/1e6:.2f} MB, {len(data['m']):,} films")
