from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import urlparse


def contains_internal_url(text: str) -> bool:
    for match in re.findall(r"https?://[^\s\"'<>]+", text or "", flags=re.IGNORECASE):
        ok, _ = validate_url_target(match)
        if not ok:
            return True
    return False


def validate_url_target(url: str) -> tuple[bool, str]:
    try:
        parsed = urlparse(url)
    except Exception as exc:
        return False, str(exc)
    if parsed.scheme not in ("http", "https"):
        return False, f"unsupported scheme: {parsed.scheme or 'none'}"
    if not parsed.hostname:
        return False, "missing hostname"
    return _validate_hostname(parsed.hostname)


def validate_resolved_url(url: str) -> tuple[bool, str]:
    return validate_url_target(url)


def _validate_hostname(hostname: str) -> tuple[bool, str]:
    try:
        infos = socket.getaddrinfo(hostname, None)
    except OSError as exc:
        return False, f"dns lookup failed: {exc}"
    for info in infos:
        ip = info[4][0]
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            continue
        if (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_multicast
            or addr.is_reserved
        ):
            return False, f"internal/private address not allowed: {ip}"
    return True, ""
