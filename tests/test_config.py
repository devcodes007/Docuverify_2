from app.config import Settings


def test_cors_origins_dev_always_includes_localhost():
    s = Settings(environment="development", cors_origins="https://example.com")
    origins = s.cors_origins_list
    assert "https://example.com" in origins
    assert "http://localhost:8080" in origins


def test_cors_origins_production_uses_only_configured_values():
    s = Settings(environment="production", cors_origins="https://docuverify.vercel.app")
    origins = s.cors_origins_list
    assert origins == ["https://docuverify.vercel.app"]
    assert "http://localhost:8080" not in origins


def test_cors_origins_production_supports_multiple_comma_separated():
    s = Settings(
        environment="production",
        cors_origins="https://docuverify.vercel.app, https://docuverify-git-main.vercel.app",
    )
    origins = s.cors_origins_list
    assert set(origins) == {"https://docuverify.vercel.app", "https://docuverify-git-main.vercel.app"}


def test_cors_origins_empty_in_production_blocks_everything():
    s = Settings(environment="production", cors_origins="")
    assert s.cors_origins_list == []
