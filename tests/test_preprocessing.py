from src.preprocessing import clean_text


def test_clean_text_returns_string():
    text = "Application crashes after login!!"

    cleaned = clean_text(text)

    assert isinstance(cleaned, str)


def test_clean_text_removes_punctuation():
    text = "Hello!!!"

    cleaned = clean_text(text)

    assert "!" not in cleaned


def test_clean_text_lowercase():
    text = "LOGIN FAILED"

    cleaned = clean_text(text)

    assert cleaned == cleaned.lower()


def test_clean_text_empty_string():
    cleaned = clean_text("")

    assert cleaned == ""
