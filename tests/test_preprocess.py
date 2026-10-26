import io

import numpy as np
import pytest
from PIL import Image

from tests.conftest import require_model


@pytest.fixture(autouse=True)
def _model_present():
    require_model()


def _png(w, h, mode="RGB"):
    buf = io.BytesIO()
    Image.new(mode, (w, h), color=120 if mode == "L" else (10, 200, 90)).save(buf, format="PNG")
    return buf.getvalue()


def test_preprocess_resizes_to_model_input():
    from app import model

    batch = model.preprocess(_png(640, 480))
    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32
    assert 0.0 <= batch.min() and batch.max() <= 255.0


def test_preprocess_handles_grayscale_and_rgba():
    from app import model

    assert model.preprocess(_png(50, 50, "L")).shape == (1, 224, 224, 3)
    assert model.preprocess(_png(50, 50, "RGBA")).shape == (1, 224, 224, 3)


def test_preprocess_rejects_garbage():
    from app import model

    with pytest.raises(ValueError):
        model.preprocess(b"\x00\x01\x02 nope")
