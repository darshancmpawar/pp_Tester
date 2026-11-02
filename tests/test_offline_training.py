"""
Unit tests for the offline training pipeline (scripts/train.py + the
ARTIFACT_DIR default in ml_core.py).

The conftest in tests/ stubs tensorflow / sklearn / joblib so we never
actually train; we monkeypatch train_model / artifacts_exist on the
imported module to drive the script's branches.

Run with:  pytest tests/test_offline_training.py -v
"""
import importlib
import os
import sys

import pytest


# ═══════════════════════════════════════════════════════════════════════
# ARTIFACT_DIR default
# ═══════════════════════════════════════════════════════════════════════
class TestArtifactDirDefault:
    def test_default_is_artifacts_subdir(self, monkeypatch):
        """With no ARTIFACT_DIR env var, ml_core defaults to ./artifacts/."""
        monkeypatch.delenv("ARTIFACT_DIR", raising=False)
        import ml_core
        importlib.reload(ml_core)
        assert os.path.basename(ml_core.ARTIFACT_DIR.rstrip("/")) == "artifacts"
        # The directory must exist after import (mkdirs on import).
        assert os.path.isdir(ml_core.ARTIFACT_DIR)

    def test_env_override_respected(self, tmp_path, monkeypatch):
        """ARTIFACT_DIR env var wins over the default."""
        monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
        import ml_core
        importlib.reload(ml_core)
        assert ml_core.ARTIFACT_DIR == str(tmp_path)


# ═══════════════════════════════════════════════════════════════════════
# scripts/train.py CLI behavior
# ═══════════════════════════════════════════════════════════════════════
@pytest.fixture
def train_mod(monkeypatch, tmp_path):
    """Reload scripts.train with stubbed dependencies."""
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    # Force a clean reload so the new env is honored by train.py's setdefault.
    for mod in ("ml_core", "scripts.train", "scripts"):
        sys.modules.pop(mod, None)
    import scripts.train as train  # noqa: E402
    yield train
    sys.modules.pop("scripts.train", None)


def _stub_db(monkeypatch, train_mod, db):
    """Replace CLIENT_DB inside the train module with a controlled fixture."""
    monkeypatch.setattr(train_mod, "CLIENT_DB", db)


def _patch_logic(monkeypatch, train_mod):
    """get_logic returns an object exposing only encoder_columns."""
    class _L:
        encoder_columns = ["weekday", "menu_items", "sub_category", "category"]
    monkeypatch.setattr(train_mod, "get_logic", lambda ck: _L())


class TestTrainScript:
    def test_skips_no_embeddings_clients(self, train_mod, monkeypatch, capsys):
        _stub_db(monkeypatch, train_mod, {
            "toasttab": {"name": "Toasttab", "dataset": None, "has_embeddings": False},
        })
        _patch_logic(monkeypatch, train_mod)
        # train_model must NOT be called.
        called = []
        monkeypatch.setattr(train_mod, "train_model",
                            lambda *a, **kw: called.append(a) or (None, 0.0))
        monkeypatch.setattr(train_mod, "artifacts_exist", lambda *a, **kw: False)

        rc = train_mod.train_one("toasttab", train_mod.CLIENT_DB["toasttab"])
        assert rc is True
        assert called == []
        assert "multiplier-only" in capsys.readouterr().out

    def test_fails_on_missing_dataset(self, train_mod, monkeypatch, capsys):
        _stub_db(monkeypatch, train_mod, {
            "tekion": {"name": "Tekion", "dataset": "/nonexistent.xlsx",
                       "has_embeddings": True},
        })
        _patch_logic(monkeypatch, train_mod)
        monkeypatch.setattr(train_mod, "train_model",
                            lambda *a, **kw: (None, 0.0))
        monkeypatch.setattr(train_mod, "artifacts_exist", lambda *a, **kw: False)

        rc = train_mod.train_one("tekion", train_mod.CLIENT_DB["tekion"])
        assert rc is False
        assert "dataset missing" in capsys.readouterr().out

    def test_skips_when_artifacts_present(self, train_mod, monkeypatch, tmp_path, capsys):
        ds = tmp_path / "fake.xlsx"
        ds.write_bytes(b"")
        _stub_db(monkeypatch, train_mod, {
            "tekion": {"name": "Tekion", "dataset": str(ds), "has_embeddings": True},
        })
        _patch_logic(monkeypatch, train_mod)
        called = []
        monkeypatch.setattr(train_mod, "train_model",
                            lambda *a, **kw: called.append(a) or (None, 0.0))
        monkeypatch.setattr(train_mod, "artifacts_exist", lambda *a, **kw: True)

        rc = train_mod.train_one("tekion", train_mod.CLIENT_DB["tekion"])
        assert rc is True
        assert called == [], "train_model should be skipped when artifacts present"
        assert "artifacts already present" in capsys.readouterr().out

    def test_force_retrains_even_if_present(self, train_mod, monkeypatch, tmp_path, capsys):
        ds = tmp_path / "fake.xlsx"
        ds.write_bytes(b"")
        _stub_db(monkeypatch, train_mod, {
            "tekion": {"name": "Tekion", "dataset": str(ds), "has_embeddings": True},
        })
        _patch_logic(monkeypatch, train_mod)
        called = []
        monkeypatch.setattr(train_mod, "train_model",
                            lambda *a, **kw: called.append(a) or (None, 0.123))
        monkeypatch.setattr(train_mod, "artifacts_exist", lambda *a, **kw: True)

        rc = train_mod.train_one("tekion", train_mod.CLIENT_DB["tekion"], force=True)
        assert rc is True
        assert len(called) == 1, "train_model must run with --force"
        assert "RMSE=0.1230" in capsys.readouterr().out

    def test_train_model_failure_returns_false(self, train_mod, monkeypatch, tmp_path, capsys):
        ds = tmp_path / "fake.xlsx"
        ds.write_bytes(b"")
        _stub_db(monkeypatch, train_mod, {
            "tekion": {"name": "Tekion", "dataset": str(ds), "has_embeddings": True},
        })
        _patch_logic(monkeypatch, train_mod)

        def boom(*a, **kw):
            raise RuntimeError("tf exploded")

        monkeypatch.setattr(train_mod, "train_model", boom)
        monkeypatch.setattr(train_mod, "artifacts_exist", lambda *a, **kw: False)

        rc = train_mod.train_one("tekion", train_mod.CLIENT_DB["tekion"])
        assert rc is False
        assert "tf exploded" in capsys.readouterr().out

    def test_main_returns_nonzero_when_any_client_fails(self, train_mod, monkeypatch):
        _stub_db(monkeypatch, train_mod, {
            "ok_client":   {"name": "X", "dataset": "/missing.xlsx", "has_embeddings": False},
            "fail_client": {"name": "Y", "dataset": "/missing.xlsx", "has_embeddings": True},
        })
        _patch_logic(monkeypatch, train_mod)
        monkeypatch.setattr(train_mod, "artifacts_exist", lambda *a, **kw: False)
        monkeypatch.setattr(train_mod, "train_model",
                            lambda *a, **kw: (None, 0.0))
        # simulate `python scripts/train.py --client all`
        monkeypatch.setattr(sys, "argv", ["train.py", "--client", "all"])
        rc = train_mod.main()
        assert rc == 1

    def test_main_unknown_client_fails_fast(self, train_mod, monkeypatch, capsys):
        _stub_db(monkeypatch, train_mod, {
            "tekion": {"name": "Tekion", "dataset": "/x.xlsx", "has_embeddings": True},
        })
        monkeypatch.setattr(sys, "argv", ["train.py", "--client", "no_such_client"])
        rc = train_mod.main()
        assert rc == 1
        assert "unknown client" in capsys.readouterr().out


# ═══════════════════════════════════════════════════════════════════════
# ml_core.load_dataset — sheet selection
# ═══════════════════════════════════════════════════════════════════════
class TestLoadDatasetSheetSelection:
    def test_clario_reads_wastage_sheet_not_default(self, monkeypatch):
        """Regression guard: clario must use sheet_name='Sheet1', not index 0.

        Sheet2 (index 0) in Clario_Wastage_dataset.xlsx is a pivot summary
        with unnamed columns; reading it would fail schema validation.
        """
        import pandas as pd
        import ml_core

        captured = {}

        def fake_read_excel(path, sheet_name=0):
            captured["sheet_name"] = sheet_name
            return pd.DataFrame({
                "date":         ["2025-01-06"],
                "menu_items":   ["rice"],
                "sub_category": ["grains"],
                "category":     ["veg"],
                "ideal_pp":     [0.25],
            })

        monkeypatch.setattr(ml_core.pd, "read_excel", fake_read_excel)

        class _MockLogic:
            def canonicalize_category(self, v):     return v
            def canonicalize_day_type(self, v):     return v
            def canonicalize_holiday_type(self, v): return v
            def canonicalize_meal_day(self, v):     return v

        ml_core.load_dataset("clario", "fake_path.xlsx", _MockLogic())

        assert captured.get("sheet_name") == "Sheet1", (
            f"Expected sheet_name='Sheet1', got {captured.get('sheet_name')!r}. "
            "Check client_database.py 'sheet' key and ml_core.load_dataset."
        )

    def test_clients_without_sheet_key_use_default_index(self, monkeypatch):
        """Clients with no 'sheet' key must fall back to sheet index 0."""
        import pandas as pd
        import ml_core

        captured = {}

        def fake_read_excel(path, sheet_name=0):
            captured["sheet_name"] = sheet_name
            return pd.DataFrame({
                "date":         ["2025-01-06"],
                "menu_items":   ["rice"],
                "sub_category": ["grains"],
                "category":     ["veg"],
                "ideal_pp":     [0.25],
            })

        monkeypatch.setattr(ml_core.pd, "read_excel", fake_read_excel)

        class _MockLogic:
            def canonicalize_category(self, v):     return v
            def canonicalize_day_type(self, v):     return v
            def canonicalize_holiday_type(self, v): return v
            def canonicalize_meal_day(self, v):     return v

        ml_core.load_dataset("tekion", "fake_path.xlsx", _MockLogic())

        assert captured.get("sheet_name") == 0, (
            f"Expected sheet_name=0 (default), got {captured.get('sheet_name')!r}."
        )


# ═══════════════════════════════════════════════════════════════════════
# app.py UX — _ensure() must NOT train at runtime anymore
# ═══════════════════════════════════════════════════════════════════════
class TestAppDoesNotTrain:
    def test_app_imports_no_training_symbols(self):
        """Sanity: app.py must not import train_model, train_gate, or clear_cache."""
        src = open("app.py").read()
        assert "train_model" not in src, "app.py should not reference train_model"
        assert "train_gate" not in src, "app.py should not reference train_gate"
        assert "clear_cache" not in src, "app.py should not reference clear_cache"
