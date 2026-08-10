"""CI-safe checks for the opt-in audio fixture generator."""

from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def test_fixture_generator_is_executable_and_documented():
    generator = FIXTURES / "generate_fixtures.sh"
    readme = FIXTURES / "README.md"

    assert generator.is_file()
    assert generator.stat().st_mode & 0o111
    assert "Common Voice" in readme.read_text(encoding="utf-8")
    assert "generated/" in readme.read_text(encoding="utf-8")


def test_no_generated_media_is_checked_in():
    generated = FIXTURES / "generated"
    assert not generated.exists()
