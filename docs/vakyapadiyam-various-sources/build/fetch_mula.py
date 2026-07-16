#!/usr/bin/env python3
"""
Fetch the mula (root) Devanagari karikas of Bhartrhari's Vakyapadiya from wisdomlib.

SCOPE NOTE (important, do not loosen):
This extractor reads ONLY Devanagari text nodes occurring inside <blockquote> elements.
On wisdomlib's verse pages that is exactly where the mula karika sits. Iyer's 1965
English translation, his commentary extracts and his exegetical notes live outside
blockquotes and are therefore never read, never stored, and never emitted.

The karikas themselves are Bhartrhari's (c. 5th c. CE) and are public domain.
The surrounding scholarly apparatus is not, and this script is deliberately built so
that it cannot pick it up.

Output: vakyapadiya_mula_deva.json  {label, href, title, lines[]}
Idempotent: re-running refreshes the corpus. Safe to re-run.
"""

import json
import os
import re
import sys
import time
import urllib.request
from html.parser import HTMLParser

BOOK = "https://www.wisdomlib.org/hinduism/book/vakyapadiya-of-bhartrihari"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "\
     "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "vakyapadiya_mula_deva.json")

DEVA = re.compile(r"[ऀ-ॿ]")


def get(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


class TocParser(HTMLParser):
    """Collect /d/docNNNN.html links whose anchor text starts with Verse or Book."""

    def __init__(self):
        super().__init__()
        self.links = []
        self._href = None
        self._buf = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            d = dict(attrs)
            h = d.get("href", "")
            if "/d/doc" in h and "vakyapadiya-of-bhartrihari" in h:
                self._href = h
                self._buf = []

    def handle_data(self, data):
        if self._href is not None:
            self._buf.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._href is not None:
            label = "".join(self._buf).strip()
            if re.match(r"^(Verse|Book)\b", label):
                href = self._href
                if href.startswith("/"):
                    href = "https://www.wisdomlib.org" + href
                self.links.append({"href": href.split("#")[0], "label": label})
            self._href = None
            self._buf = []


class VerseParser(HTMLParser):
    """Capture Devanagari text nodes that occur inside <blockquote> only."""

    def __init__(self):
        super().__init__()
        self.depth = 0
        self.lines = []
        self.title = None
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "blockquote":
            self.depth += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag == "blockquote" and self.depth > 0:
            self.depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title and self.title is None:
            self.title = data.strip()
        # Structural guard: only inside a blockquote, only if Devanagari present.
        if self.depth > 0 and DEVA.search(data):
            s = data.strip()
            if s:
                self.lines.append(s)


def main():
    sys.stderr.write("fetching table of contents...\n")
    tp = TocParser()
    tp.feed(get(BOOK))

    seen, toc = set(), []
    for l in tp.links:
        if l["href"] in seen:
            continue
        seen.add(l["href"])
        toc.append(l)

    sys.stderr.write("found %d verse links\n" % len(toc))

    out = []
    for i, t in enumerate(toc, 1):
        vp = VerseParser()
        try:
            vp.feed(get(t["href"]))
            out.append({"label": t["label"], "href": t["href"],
                        "title": vp.title, "lines": vp.lines})
        except Exception as e:
            sys.stderr.write("ERROR %s: %s\n" % (t["href"], e))
            out.append({"label": t["label"], "href": t["href"],
                        "title": None, "lines": [], "error": str(e)})
        if i % 50 == 0:
            sys.stderr.write("  %d/%d\n" % (i, len(toc)))
            sys.stderr.flush()
        time.sleep(0.15)  # be polite to the server

    payload = {
        "source": BOOK,
        "scope": "mula Devanagari karikas only (blockquote-scoped extraction)",
        "count": len(out),
        "with_lines": sum(1 for x in out if x["lines"]),
        "verses": out,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    sys.stderr.write("wrote %s (%d entries, %d with verses)\n"
                     % (OUT, len(out), payload["with_lines"]))


if __name__ == "__main__":
    main()
