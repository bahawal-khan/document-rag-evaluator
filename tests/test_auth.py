from types import SimpleNamespace as NS

import core.auth as auth


class FakeSecrets(dict):
    pass


def _patch(monkeypatch, secrets, user=None, login=True):
    monkeypatch.setattr(auth.st, "secrets", secrets, raising=False)
    monkeypatch.setattr(auth.st, "user", user or NS(is_logged_in=False), raising=False)
    if not login:
        monkeypatch.delattr(auth.st, "login", raising=False)


def test_not_configured_means_no_gate(monkeypatch):
    _patch(monkeypatch, FakeSecrets())
    assert auth.auth_configured() is False
    assert auth.require_login() is None  # returns instead of stopping the page


def test_incomplete_config_is_ignored(monkeypatch):
    _patch(monkeypatch, FakeSecrets(auth={"client_id": "x"}))
    assert auth.auth_configured() is False


def test_complete_config_is_detected(monkeypatch):
    _patch(monkeypatch, FakeSecrets(auth={"client_id": "x", "client_secret": "y", "redirect_uri": "https://a/oauth2callback"}))
    assert auth.auth_configured() is True


def test_unverified_email_is_not_an_identity(monkeypatch):
    _patch(monkeypatch, FakeSecrets(), user=NS(is_logged_in=True, email="a@b.c", email_verified=False))
    assert auth._signed_in_email() is None


def test_verified_user_is_recognised(monkeypatch):
    _patch(monkeypatch, FakeSecrets(), user=NS(is_logged_in=True, email="a@b.c", email_verified=True))
    assert auth._signed_in_email() == "a@b.c"
