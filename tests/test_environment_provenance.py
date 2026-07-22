def test_local_wheel_origins_are_normalized_without_touching_git_urls(monkeypatch):
    import scripts.check_environment as environment

    monkeypatch.setattr(environment, "version", lambda name: "1.2.3")
    lines = environment.normalized_freeze_lines(
        "local-package @ file:///" + "home/builder/wheel\n"
        "source @ git+https://example.test/source@abcdef\n"
    )

    assert lines == [
        "local-package==1.2.3",
        "source @ git+https://example.test/source@abcdef",
    ]
