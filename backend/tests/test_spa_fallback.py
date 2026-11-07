"""The web app's catch-all answers browsers, not API clients (#1083).

In Docker the built frontend is served with a catch-all, so a request for a
path that is not an API route, such as ``/generations/<id>/audio``, came back
as the app's ``index.html`` with a 200. An integration asking for audio got a
web page and no error.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app import _mount_frontend

BROWSER = {"accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}


def _client(tmp_path):
    (tmp_path / "index.html").write_text("<!doctype html><title>Voicebox</title>")
    (tmp_path / "favicon.ico").write_bytes(b"icon")
    app = FastAPI()

    @app.get("/audio/{generation_id}")
    def audio(generation_id: str):
        return {"id": generation_id}

    _mount_frontend(app, tmp_path)
    return TestClient(app)


def test_an_unknown_api_path_is_a_404_not_the_app(tmp_path):
    client = _client(tmp_path)
    for path in ["/generations/abc", "/generations/abc/audio", "/generations/abc/download"]:
        for headers in [{"accept": "*/*"}, {"accept": "application/json"}, {"accept": "audio/*"}]:
            response = client.get(path, headers=headers)
            assert response.status_code == 404, (path, headers, response.text[:80])
            assert response.json() == {"detail": "Not Found"}


def test_a_browser_still_gets_the_app_for_client_side_routes(tmp_path):
    client = _client(tmp_path)
    for path in ["/", "/voices", "/stories/abc"]:
        response = client.get(path, headers=BROWSER)
        assert response.status_code == 200, path
        assert "<!doctype html>" in response.text


def test_real_routes_and_files_are_unchanged(tmp_path):
    client = _client(tmp_path)
    assert client.get("/audio/abc", headers={"accept": "*/*"}).json() == {"id": "abc"}
    assert client.get("/favicon.ico", headers={"accept": "*/*"}).content == b"icon"
