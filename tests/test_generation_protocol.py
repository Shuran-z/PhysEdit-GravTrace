import sys

from scripts import phyediting_gen


def test_generated_scoring_defaults_do_not_use_reference_velocity(monkeypatch):
    captured = []
    monkeypatch.setattr(sys, "argv", ["phyediting_gen.py", "score", "items", "rows", "tracks", "out"])
    monkeypatch.setattr(phyediting_gen, "score", captured.append)
    phyediting_gen.main()
    assert captured[0].v0 == "agnostic"
    assert captured[0].offset_mode == "anchor"
