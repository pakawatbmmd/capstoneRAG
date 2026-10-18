"""Build the public report copy and the Firebase Hosting folder from docs/report.html.

    python examples/build_site.py && firebase deploy --only hosting

Writes
    docs/report_public.html  the report without the LLM's answer sentences (they closely follow NICE wording)
    site/index.html          full report     -> https://capstone-rag-guarded.web.app/
    site/report.html         full report     -> /report
    site/public.html         public copy     -> /public
The report is authored as a page body (title, style, content); hosted copies get a full HTML document around it.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS, SITE = ROOT / "docs", ROOT / "site"
PUBLIC_NOTE_OLD = ("Extractive pipelines are listed by the recommendation they returned rather than its full text, "
                   "to keep NICE wording out of this page.</p>")
PUBLIC_NOTE_NEW = ("Every pipeline is listed by the recommendation it returned or cited. <b>This public copy also omits "
                   "the LLM's answer sentences</b>, because they closely follow NICE wording, which may not be "
                   "redistributed outside the UK.</p>")


def public_copy(html: str) -> str:
    match = re.search(r"const QDATA=(\[.*?\]);\n", html, flags=re.S)
    questions = json.loads(match.group(1).replace("<\\/", "</"))
    for item in questions:
        for result in item["sys"].values():
            result["ans"] = ""
    html = html[: match.start(1)] + json.dumps(questions, ensure_ascii=False).replace("</", "<\\/") + html[match.end(1):]
    assert html.count(PUBLIC_NOTE_OLD) == 1
    html = html.replace(PUBLIC_NOTE_OLD, PUBLIC_NOTE_NEW)
    return html.replace("<title>Guarded Guideline Q&amp;A</title>", "<title>Guarded Guideline Q&amp;A (public)</title>", 1)


def as_document(body: str) -> str:
    assert body.lstrip().startswith("<title>") and body.count('<div id="tip"') == 1
    head = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')
    return head + body.replace('<div id="tip"', '</head>\n<body>\n<div id="tip"', 1) + "\n</body>\n</html>\n"


def main():
    full = (DOCS / "report.html").read_text(encoding="utf-8")
    public = public_copy(full)
    (DOCS / "report_public.html").write_text(public, encoding="utf-8")
    SITE.mkdir(exist_ok=True)
    for name, body in (("index.html", full), ("report.html", full), ("public.html", public)):
        (SITE / name).write_text(as_document(body), encoding="utf-8")
    print("built docs/report_public.html and site/{index,report,public}.html")


if __name__ == "__main__":
    main()
