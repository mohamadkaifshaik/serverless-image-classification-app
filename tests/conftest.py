import io
import pathlib

import pytest
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "model" / "flowers_mobilenetv2.keras"


def require_model():
    if not MODEL_FILE.exists():
        pytest.skip("no trained model in model/ - run training/train.py first")


@pytest.fixture(scope="session")
def client():
    require_model()
    from app.main import app

    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def jpeg_bytes():
    """A synthetic 300x200 image - we only care that the pipeline runs, not that it is a real flower."""
    img = Image.new("RGB", (300, 200), color=(240, 200, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()
