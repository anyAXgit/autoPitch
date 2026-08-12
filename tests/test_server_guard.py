"""A page in another tab must not be able to drive the local server.

Binding 127.0.0.1 stops other machines, not other origins. These tests drive a
real socket rather than calling the handler directly, because the whole point
is which *headers* arrive -- a mocked handler would prove nothing about what a
browser actually sends.
"""
import http.client
import threading
from http.server import ThreadingHTTPServer

import pytest

from gui.server import Handler, is_loopback


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def call(port, path="/", method="GET", host=None, **headers):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
    c.putheader("Host", host or f"127.0.0.1:{port}")
    for k, v in headers.items():
        c.putheader(k.replace("_", "-"), v)
    if method == "POST":
        c.putheader("Content-Length", "2")
    c.endheaders()
    if method == "POST":
        c.send(b"{}")
    code = c.getresponse().status
    c.close()
    return code


# ---- the hostname check, on its own ----

@pytest.mark.parametrize("value", [
    "127.0.0.1:8756", "localhost:8756", "127.0.0.1", "localhost",
    "[::1]:8756", "http://127.0.0.1:8756", "http://localhost:8756",
])
def test_loopback_accepted(value):
    assert is_loopback(value)


@pytest.mark.parametrize("value", [
    # 127.0.0.1.evil.example resolves wherever its owner points it -- the
    # suffix is what rebinding relies on, so a substring test would let it in.
    "evil.example:8756", "127.0.0.1.evil.example", "http://evil.example",
    "https://attacker.test", "null", "", "0.0.0.0", "192.168.1.5:8756",
])
def test_non_loopback_rejected(value):
    assert not is_loopback(value)


# ---- the same checks over a live socket ----

def test_own_page_is_served(server):
    assert call(server) == 200


def test_rebinding_host_refused(server):
    """The packet reached loopback, but the browser believed it was elsewhere;
    that belief is what lets the attacker's origin read the reply."""
    assert call(server, host=f"evil.example:{server}") == 403


def test_cross_origin_post_refused(server):
    assert call(server, "/api/install_ffmpeg", "POST",
                Origin="http://evil.example") == 403


def test_cross_site_image_get_refused(server):
    """<img src="http://127.0.0.1:8756/api/reveal?..."> sends no Origin at all;
    Sec-Fetch-Site is the only header that gives it away."""
    assert call(server, "/api/reveal", Sec_Fetch_Site="cross-site") == 403


def test_same_site_but_not_same_origin_refused(server):
    assert call(server, "/api/state", Sec_Fetch_Site="same-site") == 403


def test_our_own_fetch_passes(server):
    assert call(server, "/", Origin=f"http://127.0.0.1:{server}",
                Sec_Fetch_Site="same-origin") == 200


def test_curl_still_works(server):
    """No Origin and no Sec-Fetch-Site means no browser was involved: a page
    cannot make the browser omit them. Scripts and CI keep working."""
    assert call(server, "/api/__guard_probe__", "POST") == 404
