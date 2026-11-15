"""Assembles the live web version of the film into one self-contained page.

    python web_export.py     # once: world + score -> build/web/data.b64
    python build_web.py      # -> Mulberry Bush 2026 (live).html  and  build/web/artifact.html

The first file is a complete HTML document to open from disk. The second is
the same page without the document skeleton, for publishing as a claude.ai
artifact (the host adds the skeleton).
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "web")


def read(*p):
    with open(os.path.join(*p), encoding="utf8") as f:
        return f.read()


def main():
    page = read(WEB, "template.html")
    parts = {
        "DATA": read(HERE, "build", "web", "data.b64"),
        "GROUND_WORKER": read(WEB, "ground-worker.js"),
        "AUDIO_WORKER": read(WEB, "audio-worker.js"),
        "TIMELINE": read(WEB, "timeline.js"),
        "MAIN": "\n".join(read(WEB, f) for f in ("data.js", "film.js", "sound.js", "app.js")),
    }
    for k, v in parts.items():
        assert "</script" not in v.lower(), k
        n = page.count("{{%s}}" % k)
        assert n == 1, (k, n)
        page = page.replace("{{%s}}" % k, v)
    assert "{{" not in re.sub(r"\$\{[^}]*\}", "", page.split('id="data">')[0]), "unfilled placeholder"
    os.makedirs(os.path.join(HERE, "build", "web"), exist_ok=True)
    frag = os.path.join(HERE, "build", "web", "artifact.html")
    with open(frag, "w", encoding="utf8", newline="\n") as f:
        f.write(page)
    full = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')
    head, body = page.split("</style>", 1)
    full += head + "</style>\n</head>\n<body>" + body + "</body>\n</html>\n"
    out = os.path.join(HERE, "Mulberry Bush 2026 (live).html")
    with open(out, "w", encoding="utf8", newline="\n") as f:
        f.write(full)
    print("wrote", out, f"{os.path.getsize(out) / 1024:.0f} KB")
    print("wrote", frag, f"{os.path.getsize(frag) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
