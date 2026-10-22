"""API integration tests: analysis lifecycle, artifacts, reports, evaluation, deletion, security."""
import json

import numpy as np
import pytest

from tests.conftest import png_bytes


def files_for(pair):
    before, after, _ = pair
    return {"before": ("before.png", png_bytes(before), "image/png"),
            "after": ("after.png", png_bytes(after), "image/png")}


@pytest.fixture(scope="module")
def analysis(client, urban_pair):
    r = client.post("/api/analyze", files=files_for(urban_pair), data={"params": json.dumps({"method": "classical"})})
    assert r.status_code == 201, r.text
    return r.json()


def test_health_distinguishes_backend_and_model(client):
    r = client.get("/api/health").json()
    assert r["status"] == "ok" and r["backend"] == "available"
    assert r["model"]["status"] == "missing" and r["model"]["ready"] is False
    assert r["capabilities"]["classical"] is True and r["capabilities"]["deep_learning"] is False


def test_models_endpoint(client):
    r = client.get("/api/models").json()
    methods = {m["id"]: m for m in r["methods"]}
    assert methods["classical"]["available"] and not methods["classical"]["is_trained_model"]
    assert methods["deep"]["available"] is False
    assert "checkpoint" in methods["deep"]["model"]["message"].lower()


def test_classical_analysis_result(analysis, urban_pair):
    s, res = analysis["summary"], analysis["result"]
    assert s["method"] == "classical" and s["model_name"] is None
    assert res["method"]["is_trained_model"] is False
    assert "baseline" in res["method"]["label"].lower()
    st = res["statistics"]
    assert st["changed_pixel_percentage"] == pytest.approx(st["changed_pixel_count"] / st["valid_pixel_count"] * 100, abs=1e-3)
    assert 0.5 < st["changed_pixel_percentage"] < 10  # GT for this scene is ~2%
    assert res["area"]["changed_area_m2"] is None  # no georeferencing -> no physical area
    assert res["alignment"]["status"] == "aligned"
    types = {a["artifact_type"] for a in analysis["artifacts"]}
    assert {"before", "after", "mask", "overlay", "heatmap", "intensity", "valid", "results"} <= types


def test_deep_unavailable_is_rejected_not_substituted(client, urban_pair):
    r = client.post("/api/analyze", files=files_for(urban_pair), data={"params": json.dumps({"method": "deep"})})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "model_unavailable"


def test_invalid_params(client, urban_pair):
    for bad in ('{"threshold": 3}', '{"method": "magic"}', "not json", '{"unknown": 1}'):
        r = client.post("/api/analyze", files=files_for(urban_pair), data={"params": bad})
        assert r.status_code == 422, bad


def test_corrupted_and_unsupported_uploads(client, urban_pair):
    before = png_bytes(urban_pair[0])
    r = client.post("/api/analyze", files={"before": ("b.png", before, "image/png"),
                                           "after": ("a.png", before[:300], "image/png")})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_image"
    r = client.post("/api/analyze", files={"before": ("b.png", before, "image/png"),
                                           "after": ("a.txt", b"hello world", "text/plain")})
    assert r.status_code == 422


def test_upload_too_large(client, urban_pair):
    big = b"\x89PNG\r\n\x1a\n" + b"0" * (6 * 1024 * 1024)  # limit is 5 MB in tests
    r = client.post("/api/analyze", files={"before": ("b.png", big, "image/png"),
                                           "after": ("a.png", png_bytes(urban_pair[1]), "image/png")})
    assert r.status_code == 413


def test_errors_do_not_leak_paths(client):
    r = client.get("/api/analyses/" + "0" * 32)
    assert r.status_code == 404
    assert ":\\" not in r.text and "/tmp" not in r.text and "orbit_test" not in r.text


def test_history_and_detail(client, analysis):
    page = client.get("/api/analyses?page=1&page_size=5").json()
    assert page["total"] >= 1 and any(i["id"] == analysis["summary"]["id"] for i in page["items"])
    d = client.get(f"/api/analyses/{analysis['summary']['id']}").json()
    assert d["summary"]["before_filename"] == "before.png"
    # timestamps must carry an explicit UTC offset so browsers don't treat them as local time
    assert d["summary"]["created_at"].endswith(("Z", "+00:00"))
    assert page["items"][0]["created_at"].endswith(("Z", "+00:00"))


def test_artifacts_served(client, analysis):
    aid = analysis["summary"]["id"]
    arts = client.get(f"/api/analyses/{aid}/artifacts").json()
    assert any(a["artifact_type"] == "mask" for a in arts)
    for path in (f"/api/analyses/{aid}/mask", f"/api/analyses/{aid}/overlay",
                 f"/api/analyses/{aid}/artifacts/heatmap"):
        r = client.get(path)
        assert r.status_code == 200 and r.content[:8] == b"\x89PNG\r\n\x1a\n"
    r = client.get(f"/api/analyses/{aid}/artifacts/results?download=true")
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    assert json.loads(r.content)["id"] == aid
    assert client.get(f"/api/analyses/{aid}/artifacts/secret").status_code == 404


def test_report_pdf(client, analysis):
    r = client.get(f"/api/analyses/{analysis['summary']['id']}/report")
    assert r.status_code == 200
    assert r.content[:5] == b"%PDF-" and len(r.content) > 10_000
    assert r.headers["content-type"] == "application/pdf"


def test_path_traversal_and_bad_ids(client):
    for bad in ("..%2F..%2Fetc", "abc", "../../secret", "0" * 31 + "g"):
        assert client.get(f"/api/analyses/{bad}").status_code in (404, 422)
    assert client.get("/api/samples/..%2Fmanifest/before").status_code == 404
    assert client.get("/api/samples/urban_expansion/../../x").status_code == 404


def test_rethreshold_updates_stats_without_rerun(client, analysis):
    aid = analysis["summary"]["id"]
    before_pct = analysis["result"]["statistics"]["changed_pixel_percentage"]
    r = client.post(f"/api/analyses/{aid}/rethreshold", json={"threshold": 0.4})
    assert r.status_code == 200
    after_pct = r.json()["result"]["statistics"]["changed_pixel_percentage"]
    assert after_pct < before_pct
    # back to the original threshold reproduces the original numbers exactly
    r = client.post(f"/api/analyses/{aid}/rethreshold", json={"threshold": analysis["result"]["threshold"]["value"]})
    assert r.json()["result"]["statistics"]["changed_pixel_percentage"] == pytest.approx(before_pct)


def test_evaluate_with_ground_truth(client, analysis, urban_pair):
    aid = analysis["summary"]["id"]
    gt = urban_pair[2]
    r = client.post("/api/evaluate", data={"analysis_id": aid},
                    files={"ground_truth": ("gt.png", png_bytes(gt), "image/png")})
    assert r.status_code == 200, r.text
    m = r.json()["metrics"]
    assert m["f1"] is not None and 0.5 < m["f1"] <= 1.0
    assert m["tp"] + m["fp"] + m["fn"] + m["tn"] == r.json()["evaluated_pixels"]
    bad = client.post("/api/evaluate", data={"analysis_id": aid},
                      files={"ground_truth": ("gt.png", png_bytes(np.zeros((40, 40), np.uint8)), "image/png")})
    assert bad.status_code == 422


def test_samples(client):
    s = client.get("/api/samples").json()["samples"]
    if not s:
        pytest.skip("bundled samples not generated")
    assert all(x["synthetic"] for x in s)
    r = client.get(f"/api/samples/{s[0]['id']}/before")
    assert r.status_code == 200 and r.content[:4] == b"\x89PNG"


def test_batch(client, urban_pair):
    b, a, _ = urban_pair
    files = [("befores", ("b1.png", png_bytes(b), "image/png")), ("befores", ("b2.png", png_bytes(b), "image/png")),
             ("afters", ("a1.png", png_bytes(a), "image/png")), ("afters", ("a2.png", b"junk", "image/png"))]
    r = client.post("/api/analyze/batch", files=files)
    assert r.status_code == 200
    res = r.json()["results"]
    assert res[0]["ok"] is True and res[1]["ok"] is False and res[1]["error"]
    r = client.post("/api/analyze/batch", files=files[:3])
    assert r.status_code == 422


def test_delete_removes_record_and_files(client, urban_pair, tmp_root):
    r = client.post("/api/analyze", files=files_for(urban_pair))
    aid = r.json()["summary"]["id"]
    folder = tmp_root / "artifacts" / "analyses" / aid
    assert folder.is_dir() and any(folder.iterdir())
    d = client.delete(f"/api/analyses/{aid}")
    assert d.status_code == 200 and d.json()["deleted_files"] is True
    assert not folder.exists()
    assert client.get(f"/api/analyses/{aid}").status_code == 404
    assert client.get(f"/api/analyses/{aid}/mask").status_code == 404
    assert client.delete(f"/api/analyses/{aid}").status_code == 404


def test_database_persistence(client, analysis):
    from sqlalchemy import select

    from app.database import session as dbs
    from app.database.models import Analysis, Artifact
    with dbs._SessionLocal() as db:
        rec = db.get(Analysis, analysis["summary"]["id"])
        assert rec is not None and rec.valid_pixel_count > 0
        arts = db.scalars(select(Artifact).where(Artifact.analysis_id == rec.id)).all()
        assert all(not a.stored_filename.startswith(("/", ".")) for a in arts)


def test_deep_analysis_with_loaded_checkpoint(client, urban_pair, tmp_path):
    """Use an UNTRAINED, randomly initialised checkpoint purely to test the inference path."""
    from app.models.checkpoint import save_checkpoint
    from app.models.siamese_unet import FCSiamDiff
    from app.services.deep import model_service
    p = tmp_path / "rand.pt"
    save_checkpoint(p, FCSiamDiff(base_channels=8), {"model_name": "random-init", "model_version": "test",
                                                     "dataset": "synthetic-unit-test"})
    model_service.load(p, "cpu")
    try:
        assert client.get("/api/health").json()["model"]["ready"] is True
        r = client.post("/api/analyze", files=files_for(urban_pair),
                        data={"params": json.dumps({"method": "deep", "threshold": 0.5})})
        assert r.status_code == 201, r.text
        res = r.json()["result"]
        assert res["method"]["type"] == "deep" and res["method"]["is_trained_model"]
        assert res["method"]["model"]["name"] == "random-init"
        assert any("SYNTHETIC" in w for w in res["warnings"])
    finally:
        model_service.load(tmp_path / "gone.pt")
    assert client.get("/api/health").json()["model"]["ready"] is False
