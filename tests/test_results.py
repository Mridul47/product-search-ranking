import json

from esci.config import load_config
from esci.eval import results as rs


def test_record_contents(tmp_path):
    cfg = load_config()
    metrics = {"ndcg@10": 1.0, "mrr": float("nan")}
    path = rs.write_result(
        tmp_path, "oracle_judged_val", cfg, metrics, {"split": "val"}, "https://wandb.example/1"
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "oracle_judged_val.json"
    assert record["metrics"] == {"ndcg@10": 1.0, "mrr": None}
    assert record["config_fingerprint"] == cfg.fingerprint()
    assert record["params"] == {"split": "val"}
    assert record["wandb_url"] == "https://wandb.example/1"


def test_rerun_replaces_the_file(tmp_path):
    cfg = load_config()
    rs.write_result(tmp_path, "x", cfg, {"mrr": 0.5}, {}, None)
    rs.write_result(tmp_path, "x", cfg, {"mrr": 0.7}, {}, None)
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text(encoding="utf-8"))["metrics"] == {"mrr": 0.7}


def test_git_state_outside_a_repository(tmp_path):
    assert rs.git_state(tmp_path) == {"commit": None, "uncommitted_changes": None}
