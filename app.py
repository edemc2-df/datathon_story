from pathlib import Path
import runpy


ROOT_DIR = Path(__file__).resolve().parent
TARGET_APP = ROOT_DIR / "streamlit_app" / "app.py"

if not TARGET_APP.exists():
    raise FileNotFoundError(f"Arquivo Streamlit não encontrado: {TARGET_APP}")

runpy.run_path(str(TARGET_APP), run_name="__main__")
