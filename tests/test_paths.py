from faultline import paths


def test_repo_root_contains_pyproject():
    assert (paths.repo_root() / "pyproject.toml").exists()


def test_data_root_defaults_inside_repo(monkeypatch):
    monkeypatch.delenv(paths.ENV_DATA, raising=False)
    assert paths.data_root() == paths.repo_root() / "data"


def test_data_root_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_DATA, str(tmp_path))
    assert paths.data_root() == tmp_path
    assert paths.manifest_path() == tmp_path / "manifest.csv"


def test_ensure_data_dirs_creates_folders(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_DATA, str(tmp_path / "d"))
    paths.ensure_data_dirs()
    assert paths.clips_dir().is_dir()
    assert paths.degraded_dir().is_dir()
