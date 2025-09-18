# Configuration file for Sphinx documentation builder.

import os
import sys
from datetime import datetime
import importlib.metadata

# -- Path setup --------------------------------------------------------------
# Add project root (one level up) and src/ to sys.path for future autodoc usage
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
for p in (PROJECT_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# -- Project information -----------------------------------------------------
project = "LangChain Decorators"
author = "ju-bezdek"
try:
    release = importlib.metadata.version("langchain-decorators")
except importlib.metadata.PackageNotFoundError:  # package not installed yet
    release = "0.0.0"
version = release
current_year = datetime.now().year
copyright = f"{current_year}, {author}"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinxext.opengraph",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
]
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_title = project
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "light_logo": "logo.png",
    "dark_logo": "logo.png",
    "navigation_with_keys": True,
}

# -- OpenGraph ---------------------------------------------------------------
ogp_site_url = "https://langchain-decorators.readthedocs.io/"
ogp_image = "https://langchain-decorators.readthedocs.io/_static/logo.png"
ogp_description_length = 300
ogp_type = "website"
ogp_locale = "en_US"

# -- Nitpicky / future improvements -----------------------------------------
# (enable later once all refs are valid)
# nitpicky = True
