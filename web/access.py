"""Optional shared access code for the HTTPS research prototype."""

import hashlib
import hmac
import os
import secrets
import time
from urllib.parse import parse_qs, urlsplit

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse


COOKIE_NAME = "__Host-lss_access"
SESSION_SECONDS = 12 * 60 * 60
MAX_LOGIN_BYTES = 4096

LOGIN_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LungStoryShort · Team access</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#101c2d;color:#eef7fc;
font:16px system-ui,sans-serif;min-height:100vh;display:grid;place-items:center;padding:24px}
main{width:min(440px,100%);border:1px solid #30445c;background:#17283e;padding:36px;border-radius:18px}
.brand{color:#62d5e6;font-weight:700;letter-spacing:.04em}h1{font-size:27px;margin:20px 0 12px}
p{color:#b7cadb;line-height:1.55}label{display:block;margin:24px 0 8px}
input,button{width:100%;padding:13px;border-radius:8px;font:inherit}
input{background:#0f1d2e;border:1px solid #6c869e;color:#fff}
input:focus{outline:2px solid #62d5e6;outline-offset:2px}
button{margin-top:16px;border:0;background:#62d5e6;color:#0d2034;font-weight:700;cursor:pointer}
.error{color:#ffb9b9;min-height:1.5em}.note{font-size:13px;margin-bottom:0}
</style></head><body><main><div class="brand">LungStoryShort</div>
<h1>Team access</h1><p>Enter the shared access code to open the research prototype.</p>
<form method="post" action="/login"><label for="access-code">Access code</label>
<input id="access-code" name="access_code" type="password" autocomplete="current-password" required autofocus maxlength="1024">
<button type="submit">Open prototype</button></form>
<p class="error" role="alert">__ERROR__</p>
<p class="note">Your session lasts 12 hours. Use de-identified research images only.</p>
</main></body></html>"""


class GroupAccessMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.password = os.environ.get("LSS_ACCESS_PASSWORD", "").encode("utf-8")
        configured_secret = os.environ.get("LSS_SESSION_SECRET", "")
        if self.password and configured_secret and len(configured_secret.encode("utf-8")) < 32:
            raise ValueError("LSS_SESSION_SECRET must contain at least 32 bytes.")
        self.secret = configured_secret.encode("utf-8") if configured_secret else secrets.token_bytes(32)

    def signature(self, value):
        return hmac.new(self.secret, value.encode("ascii"), hashlib.sha256).hexdigest()

    def authenticated(self, request):
        cookie = request.cookies.get(COOKIE_NAME, "")
        if not cookie or len(cookie) > 256:
            return False
        try:
            expires, nonce, signature = cookie.split(".")
            payload = f"{expires}.{nonce}"
            return (int(expires) > int(time.time())
                    and hmac.compare_digest(self.signature(payload), signature))
        except (ValueError, UnicodeError, TypeError):
            return False

    async def dispatch(self, request, call_next):
        if not self.password:
            return await call_next(request)
        if request.method == "POST" and "origin" in request.headers:
            try:
                origin = urlsplit(request.headers["origin"])
                same_origin = origin.scheme in {"http", "https"} and origin.netloc.lower() == request.headers.get("host", "").lower()
            except ValueError:
                same_origin = False
            if not same_origin:
                return JSONResponse({"detail": "This request must come from the prototype website."}, status_code=403)
        if request.url.path == "/login":
            return await self.login(request)
        if self.authenticated(request):
            return await call_next(request)
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": "Sign in with the team access code."}, status_code=401)
        return RedirectResponse("/login", status_code=303)

    async def login(self, request):
        if request.method == "GET":
            if self.authenticated(request):
                return RedirectResponse("/", status_code=303)
            return HTMLResponse(LOGIN_PAGE.replace("__ERROR__", ""))
        if request.method != "POST":
            return HTMLResponse("Method not allowed", status_code=405, headers={"Allow": "GET, POST"})
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/x-www-form-urlencoded":
            return JSONResponse({"detail": "Use the access-code form."}, status_code=415)
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > MAX_LOGIN_BYTES:
                return JSONResponse({"detail": "The login request is too large."}, status_code=413)
            body.extend(chunk)
        try:
            fields = parse_qs(body.decode("utf-8"), keep_blank_values=True, strict_parsing=True, max_num_fields=1)
            supplied = fields.get("access_code", [""])[0].encode("utf-8")
        except (ValueError, UnicodeError):
            supplied = b""
        if not hmac.compare_digest(supplied, self.password):
            return HTMLResponse(LOGIN_PAGE.replace("__ERROR__", "That access code was not accepted. Please try again."), status_code=401)
        payload = f"{int(time.time()) + SESSION_SECONDS}.{secrets.token_hex(16)}"
        response = RedirectResponse("/", status_code=303)
        response.set_cookie(COOKIE_NAME, f"{payload}.{self.signature(payload)}", max_age=SESSION_SECONDS,
                            path="/", secure=True, httponly=True, samesite="strict")
        return response
