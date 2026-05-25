import json

from src.lindorm_mcp_server import utils


class _Response:
    def __init__(self):
        self._payload = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_text_embedding_uses_openai_compatible_lindormai_endpoint(monkeypatch):
    captured = {}

    def fake_post(url, data, headers, verify, timeout):
        captured["url"] = url
        captured["data"] = json.loads(data)
        captured["headers"] = headers
        captured["verify"] = verify
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(utils.requests, "post", fake_post)

    code, embeddings = utils.text_embedding(
        "ld-test-proxy-ai-pub.lindorm.aliyuncs.com",
        "root",
        "secret",
        "text-embedding-v4",
        "hello",
        use_ssl=False,
        verify_ssl=False,
        dimensions=1024,
    )

    assert code == 0
    assert embeddings == [[0.1, 0.2, 0.3]]
    assert captured["url"] == (
        "http://ld-test-proxy-ai-pub.lindorm.aliyuncs.com:9002"
        "/dashscope/compatible-mode/v1/embeddings"
    )
    assert captured["data"] == {
        "model": "text-embedding-v4",
        "input": ["hello"],
        "encoding_format": "float",
        "dimensions": 1024,
    }
    assert captured["headers"]["x-ld-ak"] == "root"
    assert captured["headers"]["x-ld-sk"] == "secret"
    assert captured["headers"]["Accept-Encoding"] == "identity"
    assert captured["verify"] is False


def test_ensure_compatible_base_url_is_idempotent():
    assert utils._ensure_compatible_base_url(
        "https://example.com/dashscope/compatible-mode/v1",
        use_ssl=True,
    ) == "https://example.com/dashscope/compatible-mode/v1"
