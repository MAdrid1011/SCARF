import pytest


torch = pytest.importorskip("torch")


def test_checkpoint_loader_records_exact_tensor_coverage():
    from integration.model_loader import load_checkpoint_state

    model = torch.nn.Linear(3, 2)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    report = load_checkpoint_state(model, {"state_dict": state})

    assert report["matched_tensors"] == len(state)
    assert report["unmatched_checkpoint_tensors"] == 0
    assert report["matched_checkpoint_numel_fraction"] == 1.0
    assert model._scarf_checkpoint_load == report


def test_checkpoint_loader_rejects_empty_or_shape_mismatched_state():
    from integration.model_loader import load_checkpoint_state

    model = torch.nn.Linear(3, 2)
    with pytest.raises(ValueError, match="no tensor matching"):
        load_checkpoint_state(model, {"state_dict": {"other": torch.ones(1)}})
    with pytest.raises(ValueError, match="shape mismatch"):
        load_checkpoint_state(
            model, {"state_dict": {"weight": torch.ones(7, 7)}}
        )


def test_encoder_only_wrapper_preserves_checkpoint_key_prefixes():
    from integration.model_loader import EncoderOnlyModel, load_checkpoint_state

    model = EncoderOnlyModel(torch.nn.Linear(3, 2))
    state = {key: value.clone() for key, value in model.state_dict().items()}
    report = load_checkpoint_state(model, {"state_dict": state})

    assert set(state) == {"encoder.weight", "encoder.bias"}
    assert report["matched_checkpoint_numel_fraction"] == 1.0


def test_checkpoint_loader_uses_runtime_torch_after_archive_shim_is_cached(monkeypatch):
    import integration.model_loader as model_loader

    class ArchiveTorch:
        pass

    monkeypatch.setattr(model_loader, "torch", ArchiveTorch)
    model = torch.nn.Linear(3, 2)
    state = {key: value.clone() for key, value in model.state_dict().items()}

    report = model_loader.load_checkpoint_state(model, {"state_dict": state})

    assert report["matched_tensors"] == len(state)


def test_depthsplat_encoder_rejects_a_foreign_preloaded_dinov2_module(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import integration.model_loader as model_loader

    foreign_origin = tmp_path / "foreign" / "dinov2" / "__init__.py"
    foreign_origin.parent.mkdir(parents=True)
    foreign_origin.write_text("fixture = True\n", encoding="utf-8")
    monkeypatch.setitem(
        model_loader.sys.modules,
        "dinov2",
        SimpleNamespace(__file__=str(foreign_origin)),
    )

    with pytest.raises(RuntimeError, match="foreign preloaded DINOv2 module"):
        model_loader.get_depthsplat_encoder(lambda _cfg: (None, None), object())


def _clear_dinov2_modules(monkeypatch, modules):
    for name in tuple(modules):
        if name == "dinov2" or name.startswith("dinov2."):
            monkeypatch.delitem(modules, name, raising=False)


def test_depthsplat_loader_accepts_a_pinned_dinov2_namespace_module(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import integration.model_loader as model_loader

    pinned_root = tmp_path / "pinned-dinov2"
    namespace_root = pinned_root / "dinov2" / "hub" / "cell_dino"
    namespace_root.mkdir(parents=True)
    _clear_dinov2_modules(monkeypatch, model_loader.sys.modules)
    monkeypatch.setattr(model_loader, "DINOV2_SOURCE", pinned_root)
    monkeypatch.setitem(
        model_loader.sys.modules,
        "dinov2.hub.cell_dino",
        SimpleNamespace(__path__=[str(namespace_root)]),
    )

    model_loader._validate_pinned_dinov2_module_origins(require_loaded=True)


def test_depthsplat_loader_rejects_a_foreign_dinov2_namespace_module(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import integration.model_loader as model_loader

    pinned_root = tmp_path / "pinned-dinov2"
    foreign_root = tmp_path / "foreign-dinov2" / "dinov2" / "hub" / "cell_dino"
    pinned_root.mkdir()
    foreign_root.mkdir(parents=True)
    _clear_dinov2_modules(monkeypatch, model_loader.sys.modules)
    monkeypatch.setattr(model_loader, "DINOV2_SOURCE", pinned_root)
    monkeypatch.setitem(
        model_loader.sys.modules,
        "dinov2.hub.cell_dino",
        SimpleNamespace(__path__=[str(foreign_root)]),
    )

    with pytest.raises(RuntimeError, match="foreign preloaded DINOv2 module"):
        model_loader._validate_pinned_dinov2_module_origins(require_loaded=True)


def _clear_classic_src_modules(monkeypatch, modules):
    for name in tuple(modules):
        if name == "src" or name.startswith("src."):
            monkeypatch.delitem(modules, name, raising=False)


def test_mvsplat_loader_rejects_a_foreign_cached_src_module(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import integration.model_loader as model_loader

    mvsplat_root = tmp_path / "mvsplat"
    foreign_origin = tmp_path / "transplat" / "src" / "__init__.py"
    foreign_origin.parent.mkdir(parents=True)
    foreign_origin.write_text("fixture = True\n", encoding="utf-8")
    _clear_classic_src_modules(monkeypatch, model_loader.sys.modules)
    monkeypatch.setitem(
        model_loader.sys.modules,
        "src",
        SimpleNamespace(__file__=str(foreign_origin)),
    )

    with pytest.raises(RuntimeError, match="foreign preloaded MVSplat src module"):
        model_loader._validate_mvsplat_src_module_origins(
            mvsplat_root, require_loaded=True
        )


def test_mvsplat_loader_accepts_its_cached_src_namespace_module(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import integration.model_loader as model_loader

    mvsplat_root = tmp_path / "mvsplat"
    namespace_root = mvsplat_root / "src" / "model" / "encoder"
    namespace_root.mkdir(parents=True)
    _clear_classic_src_modules(monkeypatch, model_loader.sys.modules)
    monkeypatch.setitem(
        model_loader.sys.modules,
        "src.model.encoder",
        SimpleNamespace(__path__=[str(namespace_root)]),
    )

    model_loader._validate_mvsplat_src_module_origins(
        mvsplat_root, require_loaded=True
    )


def test_mvsplat_loader_rejects_a_mixed_src_namespace_package(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import integration.model_loader as model_loader

    mvsplat_root = tmp_path / "mvsplat"
    mvsplat_namespace = mvsplat_root / "src"
    foreign_namespace = tmp_path / "transplat" / "src"
    mvsplat_namespace.mkdir(parents=True)
    foreign_namespace.mkdir(parents=True)
    _clear_classic_src_modules(monkeypatch, model_loader.sys.modules)
    monkeypatch.setitem(
        model_loader.sys.modules,
        "src",
        SimpleNamespace(__path__=[str(mvsplat_namespace), str(foreign_namespace)]),
    )

    with pytest.raises(RuntimeError, match="foreign preloaded MVSplat src module"):
        model_loader._validate_mvsplat_src_module_origins(
            mvsplat_root, require_loaded=True
        )
