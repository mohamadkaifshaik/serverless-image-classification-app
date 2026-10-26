import base64
import io
import json

FLOWERS = {"daisy", "dandelion", "roses", "sunflowers", "tulips"}


def _multipart(data: bytes):
    return dict(data={"file": (io.BytesIO(data), "x.jpg")}, content_type="multipart/form-data")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok"
    assert set(body["classes"]) == FLOWERS


def test_predict_multipart(client, jpeg_bytes):
    r = client.post("/predict", **_multipart(jpeg_bytes))
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body["label"] in FLOWERS
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["predictions"]) == 3
    confs = [p["confidence"] for p in body["predictions"]]
    assert confs == sorted(confs, reverse=True)
    assert isinstance(body["inference_ms"], int)


def test_predict_base64_json(client, jpeg_bytes):
    payload = {"image": "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode()}
    r = client.post("/predict?top_k=5", data=json.dumps(payload), content_type="application/json")
    assert r.status_code == 200, r.get_json()
    assert len(r.get_json()["predictions"]) == 5


def test_predict_no_image(client):
    r = client.post("/predict")
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_predict_garbage_bytes(client):
    r = client.post("/predict", **_multipart(b"definitely not an image"))
    assert r.status_code == 400
    assert "decode" in r.get_json()["error"]


def test_predict_bad_base64(client):
    r = client.post("/predict", data=json.dumps({"image": "!!!not base64!!!"}), content_type="application/json")
    assert r.status_code == 400


def test_404_is_json(client):
    r = client.get("/nope")
    assert r.status_code == 404
    assert r.get_json()["error"] == "not found"
