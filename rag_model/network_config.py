import os
from urllib.parse import urlparse

DEAD_PROXY_HOSTS = {"127.0.0.1", "localhost", "::1"}
DEAD_PROXY_PORT = 9
PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "GIT_HTTP_PROXY",
    "GIT_HTTPS_PROXY",
)


def _is_dead_local_proxy(proxy_value: str) -> bool:
    normalized = proxy_value.strip()
    if "://" not in normalized:
        normalized = f"http://{normalized}"

    parsed = urlparse(normalized)
    return parsed.hostname in DEAD_PROXY_HOSTS and parsed.port == DEAD_PROXY_PORT


def sanitize_dead_local_proxies() -> list[str]:
    """
    Remove proxy variables that point to the known dead localhost:9 sink.

    This preserves legitimate proxy settings while recovering from the common
    "disable all outbound traffic" environment override that breaks Groq and
    Hugging Face requests.
    """
    removed = []

    for env_var in PROXY_ENV_VARS:
        value = os.getenv(env_var)
        if value and _is_dead_local_proxy(value):
            os.environ.pop(env_var, None)
            removed.append(env_var)

    return removed
