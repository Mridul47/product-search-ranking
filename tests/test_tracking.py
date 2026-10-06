import sys

from esci.config import load_config
from esci.eval import tracking


class _FakeRun:
    url = "https://wandb.example/run/1"

    def __init__(self):
        self.logged = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    def log(self, data):
        self.logged.append(data)


class _FakeWandb:
    def __init__(self):
        self.calls = []
        self.run = _FakeRun()

    def init(self, **kwargs):
        self.calls.append(kwargs)
        return self.run


class _ForbiddenWandb:
    def init(self, **kwargs):
        raise AssertionError("wandb must not be called when tracking is disabled")


def test_log_run_sends_config_metrics_and_tags(monkeypatch):
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    cfg = load_config()

    url = tracking.log_run(
        cfg,
        "oracle_judged_val",
        {"ndcg@10": 1.0},
        params={"split": "val"},
        tags=["phase3"],
    )

    call = fake.calls[0]
    assert call["project"] == cfg.project.name
    assert call["name"] == "oracle_judged_val"
    assert call["config"]["config_fingerprint"] == cfg.fingerprint()
    assert call["config"]["split"] == "val"
    assert call["tags"] == ["phase3"]
    assert fake.run.logged == [{"ndcg@10": 1.0}]
    assert fake.run.closed
    assert url == fake.run.url


def test_disabled_tracking_does_not_call_wandb(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", _ForbiddenWandb())
    assert tracking.log_run(load_config(), "x", {"ndcg@10": 1.0}, enabled=False) is None


def test_parameters_exclude_paths():
    params = tracking.run_parameters(load_config())
    assert "paths" not in params
    assert not any("dir" in key for key in params)


def test_extra_parameters_are_added():
    params = tracking.run_parameters(load_config(), {"setup": "random"})
    assert params["setup"] == "random"
