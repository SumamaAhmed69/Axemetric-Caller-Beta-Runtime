from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "runtime" / "chatterbox_bridge.py"
text = path.read_text("utf-8")
text = text.replace(
    'DEVICE = "cuda" if torch.cuda.is_available() else "cpu"',
    '''REQUESTED_DEVICE = os.getenv("AXEMETRIC_CHATTERBOX_DEVICE", "auto").strip().lower()\nif REQUESTED_DEVICE == "auto":\n    if torch.cuda.is_available():\n        DEVICE = "cuda"\n    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():\n        DEVICE = "mps"\n    else:\n        DEVICE = "cpu"\nelif REQUESTED_DEVICE == "mps":\n    DEVICE = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"\nelif REQUESTED_DEVICE == "cuda":\n    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"\nelse:\n    DEVICE = "cpu"'''
)
text = text.replace(
    '''    if os.name == "nt":\n        return Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "Axemetric Caller"\n    return Path.home() / ".local" / "share" / "axemetric-caller"''',
    '''    if os.name == "nt":\n        return Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "Lineborn"\n    if __import__("sys").platform == "darwin":\n        return Path.home() / "Library" / "Application Support" / "Lineborn"\n    return Path.home() / ".local" / "share" / "lineborn"'''
)
text = text.replace('app = FastAPI(title="Dialforge Chatterbox"', 'app = FastAPI(title="Lineborn Chatterbox"')
text = text.replace('"owned_by": "dialforge-local"', '"owned_by": "lineborn-local"')
path.write_text(text, encoding="utf-8")
print("Patched Chatterbox bridge for macOS MPS/CPU and platform-native data paths")
