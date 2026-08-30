# -*- coding: utf-8 -*-
"""登入憑證與部署安全設定測試。"""

from pathlib import Path

from main import AUTH_STORAGE_KEY, _auth_token


def test_remembered_login_token():
    token = _auth_token("correct horse battery staple")
    assert token == _auth_token("correct horse battery staple")
    assert token != _auth_token("different password")
    assert "correct horse" not in token
    assert len(token) == 64
    assert AUTH_STORAGE_KEY.startswith("quant_app.")


def test_render_requires_password_setting():
    render_yaml = Path("render.yaml").read_text(encoding="utf-8")
    assert "APP_PASSWORD" in render_yaml
    assert "sync: false" in render_yaml
