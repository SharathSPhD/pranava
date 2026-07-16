#!/usr/bin/env python3
"""
Build the three Vakyapadiya kanda PDFs.

Layers, and where each comes from:
  1. Mula Devanagari karika  -- Bhartrhari, c. 5th c. CE. Public domain.
                                Scraped by fetch_mula.py (blockquote-scoped).
  2. IAST transliteration    -- generated mechanically from layer 1 by
                                indic_transliteration. Deterministic; a rule-based
                                transform of a public-domain text.
  3. Translation + commentary -- original work, written from the Sanskrit of layer 1.
                                Stored in commentary/kanda{1,2,3}.json.

No third-party translation or commentary is read, stored, or emitted anywhere in
this pipeline. Layer 3 is ours; if it is absent for a verse, the verse simply
prints with layers 1-2 only. That makes the build resumable: commentary accretes
verse by verse and the PDFs stay valid at every intermediate state.

Usage:  python3 make_pdf.py [1|2|3|all]
"""

import json
import os
import re
import subprocess
import sys
import tempfile

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CORPUS = os.path.join(HERE, "vakyapadiya_mula_deva.json")
COMMDIR = os.path.join(HERE, "commentary")

FONTDIR = "/usr/share/fonts/truetype/google-fonts/"
DEVAFONT = "Poppins-Regular.ttf"

KANDA_NAMES = {
    "1": ("Brahmakanda", "Brahmakāṇḍa", "On Brahman as Word"),
    "2": ("Vakyakanda", "Vākyakāṇḍa", "On the Sentence"),
    "3": ("Padakanda", "Padakāṇḍa", "On the Word"),
}

# Kanda III is organised into fourteen samuddesas; its verse labels are 3.N.M
# where N is the samuddesa. Deriving the section heading from N means every
# page gets its structural heading immediately, whether or not commentary for
# it has been written yet.
SAMUDDESA = {
    1: "Jāti-samuddeśa — On the Universal",
    2: "Dravya-samuddeśa — On Substance",
    3: "Sambandha-samuddeśa — On Relation",
    4: "Dravyalakṣaṇa-samuddeśa — On the Definition of Substance",
    5: "Guṇa-samuddeśa — On Quality",
    6: "Dik-samuddeśa — On Position",
    7: "Sādhana-samuddeśa — On the Means",
    8: "Kriyā-samuddeśa — On Action",
    9: "Kāla-samuddeśa — On Time",
    10: "Puruṣa-samuddeśa — On Person",
    11: "Saṅkhyā-samuddeśa — On Number",
    12: "Upagraha-samuddeśa — On Aspect",
    13: "Liṅga-samuddeśa — On Gender",
    14: "Vṛtti-samuddeśa — On Complex Formation",
}


def auto_section(label):
    """Structural heading derived from the label itself (Kanda III only)."""
    m = re.match(r"^Verse 3\.(\d+)\.", label)
    if m:
        return SAMUDDESA.get(int(m.group(1)))
    return None


def esc(s):
    """Escape LaTeX specials in non-Devanagari text."""
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
           "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
           "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    for k, v in rep.items():
        s = s.replace(k, v)
    # *word* markers in commentary become italics
    s = re.sub(r"\*([^*]+)\*", r"\\emph{\1}", s)
    return s


def load_corpus():
    with open(CORPUS, encoding="utf-8") as f:
        return json.load(f)["verses"]


def load_commentary(k):
    p = os.path.join(COMMDIR, "kanda%s.json" % k)
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def verse_key(label):
    m = re.match(r"^Verse (\d+)\.([\d.]+)", label)
    return (m.group(1), m.group(2)) if m else (None, None)


def sortkey(label):
    _, num = verse_key(label)
    if not num:
        return ()
    return tuple(int(x) for x in num.split(".") if x.isdigit())


PREAMBLE = r"""\documentclass[11pt,twoside]{book}
\usepackage{fontspec}
\usepackage[margin=1.05in,top=1.1in,bottom=1.15in]{geometry}
\newfontfamily\deva[Script=Devanagari,Path=%(fontdir)s,UprightFont=%(devafont)s]{Poppins}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage{needspace}
\usepackage{parskip}
\usepackage[hidelinks]{hyperref}
\setmainfont{Lora}[Path=%(fontdir)s,UprightFont=Lora-Variable.ttf,ItalicFont=Lora-Italic-Variable.ttf]
\newcommand{\iast}[1]{{\itshape #1}}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[LE,RO]{\small\thepage}
\fancyhead[RE]{\small\scshape V\=akyapad\=\i ya}
\fancyhead[LO]{\small\scshape \leftmark}
\renewcommand{\headrulewidth}{0.2pt}
\titleformat{\section}{\large\scshape}{}{0pt}{}
\setlength{\emergencystretch}{3em}
\begin{document}
"""


def build_tex(k, verses, comm):
    tex = [PREAMBLE % {"fontdir": FONTDIR, "devafont": DEVAFONT}]
    _, name_uni, subtitle = KANDA_NAMES[k]

    n_comm = sum(1 for v in verses if comm.get(v["label"], {}).get("translation"))

    tex.append(r"\thispagestyle{empty}")
    tex.append(r"\begin{center}")
    tex.append(r"{\Huge\scshape V\=akyapad\=\i ya}\\[0.3em]")
    tex.append(r"{\large of Bhart\d{r}hari}\\[2.2em]")
    tex.append(r"{\LARGE %s}\\[0.4em]" % esc(name_uni))
    tex.append(r"{\large\itshape %s}\\[3em]" % esc(subtitle))
    tex.append(r"{\large K\=a\d{n}\d{d}a %s}\\[1em]" % k)
    tex.append(r"\end{center}")
    tex.append(r"\vfill")
    tex.append(r"\noindent\rule{\textwidth}{0.4pt}")
    tex.append(r"\footnotesize")
    tex.append(r"\noindent\textbf{On this edition.}\\")
    tex.append(esc(
        "The Sanskrit karikas presented here are Bhartrhari's, composed around the "
        "fifth century CE and long in the public domain. The Roman transliteration "
        "is generated mechanically from the Devanagari according to the IAST scheme; "
        "it involves no editorial judgement."))
    tex.append(r"\\[0.5em]")
    tex.append(esc(
        "The English translations and the commentary are original to this edition. "
        "They were written from the Sanskrit and are not derived from, and do not "
        "reproduce, any published translation or commentary. They are offered as a "
        "working study aid, not as established scholarship: the Vakyapadiya is an "
        "exceptionally compressed text, and any reading of it is contestable. Where "
        "the Sanskrit is genuinely ambiguous, or where our rendering is uncertain, "
        "this is flagged in the commentary rather than smoothed over. Readers doing "
        "serious work should consult the critical editions and the published "
        "scholarly translations."))
    tex.append(r"\\[0.5em]")
    tex.append(esc("Commentary coverage in this kanda: %d of %d verses."
                   % (n_comm, len(verses))))
    tex.append(r"\normalsize")
    tex.append(r"\clearpage")
    tex.append(r"\tableofcontents")
    tex.append(r"\clearpage")
    tex.append(r"\markboth{K\=a\d{n}\d{d}a %s}{}" % k)

    cur_section = None
    for v in verses:
        label = v["label"]
        c = comm.get(label, {})

        sec = c.get("section") or auto_section(label)
        if sec and sec != cur_section:
            cur_section = sec
            tex.append(r"\clearpage\section*{%s}" % esc(sec))
            tex.append(r"\addcontentsline{toc}{section}{%s}" % esc(sec))

        # Keep a verse's three layers together: a page break between the mula
        # Devanagari and its own transliteration would be a real defect, since
        # the layers are only intelligible as a unit. The commentary may flow.
        n_lines = len(v["lines"])
        tex.append(r"\needspace{%d\baselineskip}" % (3 + 2 * n_lines + 2))
        tex.append(r"\noindent\textbf{%s}\par\vspace{0.35em}" % esc(label))

        # Layer 1: mula Devanagari
        tex.append(r"{\deva\large")
        tex.append(r"\\".join(l for l in v["lines"]))
        tex.append(r"}\par\vspace{0.45em}")

        # Layer 2: mechanical IAST
        iast = [transliterate(l, sanscript.DEVANAGARI, sanscript.IAST)
                for l in v["lines"]]
        tex.append(r"\iast{%s}\par\vspace{0.5em}" % r"\\".join(esc(x) for x in iast))

        # Layer 3: our translation + commentary
        if c.get("translation"):
            tex.append(r"\noindent %s\par\vspace{0.4em}" % esc(c["translation"]))
        if c.get("commentary"):
            tex.append(r"{\small\noindent %s\par}" % esc(c["commentary"]))
        if c.get("contested"):
            tex.append(r"{\small\itshape\noindent [Contested: %s]\par}"
                       % esc(c["contested"]))
        tex.append(r"\vspace{1.1em}")

    tex.append(r"\end{document}")
    return "\n".join(x for x in tex if x != "")


def build(k, corpus):
    verses = [v for v in corpus
              if verse_key(v["label"])[0] == k and v.get("lines")]
    verses.sort(key=lambda v: sortkey(v["label"]))
    if not verses:
        print("kanda %s: no verses" % k)
        return
    comm = load_commentary(k)

    # Guard: wisdomlib combines some karikas onto one page ("Verse 1.16-17").
    # Commentary keyed to a label the corpus does not have would be silently
    # dropped from the PDF. Fail loudly instead.
    labels = {v["label"] for v in verses}
    orphans = [key for key in comm
               if key.startswith("Verse ") and key not in labels
               and verse_key(key)[0] == k]
    if orphans:
        raise SystemExit(
            "kanda %s: commentary written for labels absent from the corpus, "
            "which would be dropped silently: %s" % (k, ", ".join(sorted(orphans))))

    tex = build_tex(k, verses, comm)

    with tempfile.TemporaryDirectory() as td:
        tp = os.path.join(td, "k.tex")
        with open(tp, "w", encoding="utf-8") as f:
            f.write(tex)
        for _ in range(2):  # twice, for the ToC
            subprocess.run(["xelatex", "-interaction=nonstopmode", "k.tex"],
                           cwd=td, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        pdf = os.path.join(td, "k.pdf")
        if not os.path.exists(pdf):
            log = os.path.join(td, "k.log")
            if os.path.exists(log):
                print(open(log, encoding="utf-8", errors="replace").read()[-3000:])
            raise SystemExit("xelatex failed for kanda %s" % k)
        name_ascii, _, _ = KANDA_NAMES[k]
        out = os.path.join(ROOT, "Vakyapadiya_Kanda_%s_%s.pdf" % (k, name_ascii))
        with open(pdf, "rb") as s, open(out, "wb") as d:
            d.write(s.read())
        n_comm = sum(1 for v in verses if comm.get(v["label"], {}).get("translation"))
        print("kanda %s -> %s  (%d verses, %d with commentary)"
              % (k, os.path.basename(out), len(verses), n_comm))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    corpus = load_corpus()
    for k in (["1", "2", "3"] if which == "all" else [which]):
        build(k, corpus)
