from composer_clf.data.download import next_link


def test_next_link_found():
    header = '<https://hf.co/api/x?cursor=abc>; rel="next"'
    assert next_link(header) == "https://hf.co/api/x?cursor=abc"


def test_next_link_missing():
    assert next_link(None) is None
    assert next_link('<https://hf.co/a>; rel="prev"') is None


def test_fetch_downloads_then_skips_existing(tmp_path):
    from composer_clf.data.download import fetch

    src = tmp_path / "src.bin"
    src.write_bytes(b"abc")
    dest = tmp_path / "out" / "copy.bin"
    assert fetch(src.as_uri(), dest).read_bytes() == b"abc"
    src.write_bytes(b"changed")
    assert fetch(src.as_uri(), dest).read_bytes() == b"abc"
    assert not dest.with_name("copy.bin.part").exists()


def test_fetch_retries_rate_limited_requests(tmp_path):
    import io
    import urllib.error

    from composer_clf.data.download import fetch

    calls, waits = [], []

    def opener(url):
        calls.append(url)
        if len(calls) < 3:
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", {"Retry-After": "2"}, None)
        return io.BytesIO(b"ok")

    dest = fetch("https://example.org/f", tmp_path / "f", opener=opener, sleep=waits.append)
    assert dest.read_bytes() == b"ok"
    assert len(calls) == 3 and waits == [2.0, 2.0]
