# Saṃsādhanī integration — findings (2026-07-16)

The local `samsaadhanii` Docker container (Amba Kulkarni's Sanskrit Computational Toolkit,
`slm1/samsaadhanii:patched`, arm64) is running and serves Apache on `localhost:8090`.

## What works
- **Morphological analyzer** `cgi-bin/scl/MT/prog/morph/morph.cgi` — self-contained, returns
  structured JSON per word. Verified:
  `?morfword=rAmaH&encoding=WX&outencoding=IAST&mode=json` →
  two analyses (verb rā / noun rāma) with full feature bundles (prayoga, lakāra, liṅga, vibhakti,
  vacana, gaṇa …). This is the authoritative śābdabodha-tradition morphology.
  - CGI was **disabled** in the container's Apache config; enabled via
    `a2enconf serve-cgi-bin && a2enmod cgi && apachectl graceful` (reversible).
- **Transliteration**: I use `indic_transliteration.sanscript` (Devanāgarī→WX) locally to feed the
  WX-encoded analyzer, avoiding dependence on the container's transliteration CGI.

## What does NOT work (honest limitation — no fabrication)
- **Sandhi/compound segmenter** `sandhi_splitter.cgi` and the full **dependency/kāraka parser**
  return empty (HTTP 200, 0 bytes). Apache error log:
  `sh: 1: /usr/lib/cgi-bin///SKT/sktgraph2: not found`.
  The INRIA **Heritage segmenter backend** (`/SKT/sktgraph2`) that these depend on is **not
  installed** in this image. So full śābdabodha dependency parsing is blocked on a missing binary.

## Consequence for milestones
- **M2 rescoped honestly** → *Morphological analysis pipeline*: whitespace/pāda-tokenize the mūla,
  transliterate to WX, query the working morph.cgi, persist analyses, and report the **true**
  coverage (fraction of tokens receiving ≥1 valid analysis). No invented parses.
- **M2b (new, blocked)** → *Full dependency/kāraka parse*: requires a working segmenter. Options:
  (a) install Heritage `sktgraph2` in the container (heavyweight, arm64 risk);
  (b) `vidyut-cheda` (Rust, pip `vidyut==0.4.0` installed; needs the ~GB vidyut-data kosha);
  (c) INRIA Heritage public API (network, external dependency).
  Decision deferred until M2 coverage tells us how much segmentation actually gains us.

## Note
`vidyut.lipi` works standalone (no data) — usable as a second transliteration check.
`vidyut.cheda.Chedaka` and `vidyut.prakriya` need the downloaded data kosha (not yet fetched).
