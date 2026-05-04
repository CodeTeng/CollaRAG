"""Smoke test: the chimera_rag package is importable and exposes __version__."""


def test_package_is_importable():
    import chimera_rag  # noqa: F401


def test_package_has_version():
    import chimera_rag

    assert hasattr(chimera_rag, "__version__")
    assert isinstance(chimera_rag.__version__, str)
    assert len(chimera_rag.__version__) > 0
