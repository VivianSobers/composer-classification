from composer_clf.data.download import next_link


def test_next_link_found():
    header = '<https://hf.co/api/x?cursor=abc>; rel="next"'
    assert next_link(header) == "https://hf.co/api/x?cursor=abc"


def test_next_link_missing():
    assert next_link(None) is None
    assert next_link('<https://hf.co/a>; rel="prev"') is None
