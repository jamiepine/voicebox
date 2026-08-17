"""HTTP header helpers shared by the download routes.

Lives here rather than in ``app`` so route modules can use it without
importing the application module — ``app`` builds the FastAPI instance at
import time, which registers those same routers and closes an import cycle.
"""

from urllib.parse import quote


def safe_content_disposition(disposition_type: str, filename: str) -> str:
    """Build a Content-Disposition header safe for non-ASCII filenames.

    Uses RFC 5987 ``filename*`` parameter so browsers can decode UTF-8
    filenames while the ``filename`` fallback stays ASCII-only.
    """
    ascii_name = "".join(c for c in filename if c.isascii() and (c.isalnum() or c in " -_.")).strip() or "download"
    utf8_name = quote(filename, safe="")
    return f"{disposition_type}; filename=\"{ascii_name}\"; filename*=UTF-8''{utf8_name}"
