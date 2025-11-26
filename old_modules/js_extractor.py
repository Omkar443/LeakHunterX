import re
import js2py
from urllib.parse import urljoin
from .utils import is_valid_url, normalize_url


class JSExtractor:
    """
    Extracts endpoints and possibly sensitive URLs inside JavaScript code
    using both Regex and AST parsing for better detection.
    """

    URL_REGEX = re.compile(
        r"""
        (?:"|')                                  # Start quotes
        (
            (?:\/|https?:\/\/)                   # Relative or absolute
            [^"'\s]+?                            # URL body
            (?:\?.*?)?                           # Optional query
        )
        (?:"|')                                  # End quotes
        """,
        re.VERBOSE | re.IGNORECASE,
    )

    API_HINTS = [
        "api", "auth", "v1", "v2",
        "token", "secret", "jwt",
        "login", "user", "admin"
    ]

    def __init__(self, base_url=None):
        self.base_url = base_url

    async def extract(self, js_content: str):
        if not js_content:
            return []

        urls = set()

        # Regex Extraction
        regex_urls = re.findall(self.URL_REGEX, js_content)
        for u in regex_urls:
            full = urljoin(self.base_url, u) if self.base_url else u
            full = normalize_url(full)
            if is_valid_url(full):
                urls.add(full)

        # AST Parsing Extraction
        try:
            ast = js2py.parse_js(js_content)
            self._extract_from_ast(ast, urls)
        except Exception:
            # AST parsing can fail on weird obfuscation — ignore safely
            pass

        return sorted(urls)

    def _extract_from_ast(self, node, result_set):
        """
        Recursively walk AST nodes to find strings containing URLs.
        """
        if isinstance(node, dict):
            for key, value in node.items():
                self._extract_from_ast(value, result_set)

                # Detect strings
                if key == "value" and isinstance(value, str):
                    for hint in self.API_HINTS:
                        if hint in value.lower():
                            if is_valid_url(value):
                                result_set.add(value)

        elif isinstance(node, list):
            for item in node:
                self._extract_from_ast(item, result_set)
