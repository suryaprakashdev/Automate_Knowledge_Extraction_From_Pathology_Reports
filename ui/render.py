"""Loads CSS and Jinja2 HTML templates from ui/ so app.py stays free of
inline markup strings. Autoescaping is on for all templates, so any
untrusted text (LLM answers, filenames, OCR'd content) passed into a
template is HTML-escaped by default."""

from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape

UI_DIR = Path(__file__).parent

_env = Environment(
    loader=FileSystemLoader(str(UI_DIR / "templates")),
    autoescape=select_autoescape(["html"]),
)


def load_css() -> str:
    return (UI_DIR / "styles.css").read_text(encoding="utf-8")


def render(template_name: str, **context) -> str:
    return _env.get_template(template_name).render(**context)
