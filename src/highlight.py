"""
Finds the sentences an answer relied on and highlights them on the original PDF page.

How it works:
1. Split the cited chunk into sentences (or table rows) and pick the ones that share the
   most words with the answer's claims that cite this chunk.
2. On the chunk's PDF pages, find those sentences word by word (5-word sequences), so line
   breaks, hyphenation and markdown symbols in the chunk don't matter.
3. Draw yellow highlights over the matching words and render the page as an image.
"""
import re

import pymupdf

STOPWORDS = set("""the a an and or of to in on for with by from as at is are was were be been its it their our
we us this that these those which such other any all also into than has have had not but if so may
could would can will""".split())

SHINGLE = 5            # consecutive words that must match for a highlight
MIN_RUN_SHARE = 0.6    # a highlighted stretch must cover this share of the sentence
MAX_SENTENCES = 3      # sentences highlighted per source
MIN_OVERLAP = 0.3      # share of a claim's key words a sentence must contain


def tokens(text):
    """Lowercase words and numbers; '63,325' -> '63325' so numbers match PDF text too."""
    return re.findall(r"[a-z0-9]+", text.lower().replace(",", ""))


def key_words(text, ignore=()):
    return {t for t in tokens(text) if t not in STOPWORDS and t not in ignore and (len(t) > 2 or t.isdigit())}


def overlap(claim_words, sentence_words):
    """Share of the claim's key words found in the sentence. Numbers match when one is a
    rounded form of the other: '$63.3 billion' (633) matches '63,325' in a table (63325)."""
    found = 0
    for w in claim_words:
        if w in sentence_words or (w.isdigit() and len(w) >= 2 and
                                   any(s.isdigit() and s.startswith(w) for s in sentence_words)):
            found += 1
    return found / len(claim_words)


def is_figure(token):
    # amounts and percentages, not years: "110", "633", "74" yes; "2026" no
    return token.isdigit() and len(token) >= 2 and not re.fullmatch(r"(19|20)\d\d", token)


def figure_bonus(claim_words, sentence_words):
    """A sentence holding the claim's exact figure is very likely its evidence."""
    figures = [w for w in claim_words if is_figure(w)]
    for f in figures:
        if f in sentence_words or any(s.isdigit() and s.startswith(f) for s in sentence_words):
            return 0.5
    return 0.0


# A sentence ends at ";" or a line break, or at "." followed by a capital letter or digit,
# but not after abbreviations such as "U.S." or "Inc."
SENTENCE_END = re.compile(
    r"(?<=;)\s+|\n+|(?<![A-Z]\.[A-Z]\.)(?<!\bInc\.)(?<!\bNo\.)(?<!\bCo\.)(?<=\.)\s+(?=[A-Z0-9(\"])"
)


def split_sentences(chunk_text):
    # markdown table rows are one "sentence" each (they are on their own line)
    parts = SENTENCE_END.split(chunk_text)
    return [p.strip(" |#*-") for p in parts if len(tokens(p)) >= SHINGLE]


def supporting_sentences(chunk_text, claims, company=""):
    """The chunk sentences that best match the claims citing this chunk."""
    ignore = set(tokens(company))   # filings say "we", not the company name
    claim_words = [key_words(c, ignore) for c in claims]
    claim_words = [cw for cw in claim_words if cw]
    if not claim_words:
        return []
    scored = []
    for sentence in split_sentences(chunk_text):
        words = key_words(sentence)
        best = max(overlap(cw, words) + figure_bonus(cw, words) for cw in claim_words)
        if best >= MIN_OVERLAP:
            scored.append((best, sentence))
    if not scored:
        return []
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[0][0]
    # keep only sentences nearly as good as the best one
    return [s for score, s in scored[:MAX_SENTENCES] if score >= 0.75 * top]


def sentence_word_hits(page_words, sentences):
    """
    Indexes of page words where a whole sentence appears. Words are matched in 5-word
    sequences, and only long unbroken stretches are kept, so a different sentence that merely
    shares a phrase ("domestic package volumes, higher fuel surcharges") is not highlighted.
    """
    # page_words: (x0, y0, x1, y1, text, ...); one PDF word can hold several tokens ("$63,325")
    flat = []   # (token, word_index)
    for idx, w in enumerate(page_words):
        for tok in tokens(w[4]):
            flat.append((tok, idx))

    hits = set()
    for s in sentences:
        t = tokens(s)
        if len(t) < SHINGLE:
            continue
        wanted = {tuple(t[i:i + SHINGLE]) for i in range(len(t) - SHINGLE + 1)}
        matched = [False] * len(flat)
        for i in range(len(flat) - SHINGLE + 1):
            if tuple(tok for tok, _ in flat[i:i + SHINGLE]) in wanted:
                for k in range(i, i + SHINGLE):
                    matched[k] = True
        # keep unbroken stretches long enough to be this sentence, not a shared phrase
        min_len = max(SHINGLE, int(len(t) * MIN_RUN_SHARE))
        start = None
        for i, m in enumerate(matched + [False]):
            if m and start is None:
                start = i
            elif not m and start is not None:
                if i - start >= min_len:
                    hits.update(idx for _, idx in flat[start:i])
                start = None
    return hits


def table_row_hits(page_words, row):
    """
    A table row such as 'Total revenue | 63,325 | 56,654': PDFs store tables column by column,
    so find the row label, then the row's values on the same line of the page.
    """
    cells = [c.strip() for c in row.split("|") if c.strip()]
    label = tokens(cells[0]) if cells else []
    values = {t for c in cells[1:] for t in tokens(c)}
    if not label:
        return set()

    hits = set()
    n = len(label)
    for i in range(len(page_words) - n + 1):
        if [t for w in page_words[i:i + n] for t in tokens(w[4])] != label:
            continue
        y0, y1 = page_words[i][1], page_words[i][3]
        mid = (y0 + y1) / 2
        hits.update(range(i, i + n))
        for j, w in enumerate(page_words):   # values on the same line as the label
            if w[1] <= mid <= w[3] and any(t in values for t in tokens(w[4])):
                hits.add(j)
    return hits


def matching_word_boxes(page_words, sentences):
    rows = [s for s in sentences if "|" in s]
    text = [s for s in sentences if "|" not in s]
    hits = sentence_word_hits(page_words, text) if text else set()
    for row in rows:
        hits |= table_row_hits(page_words, row)
    return hits


def highlighted_pages(pdf_path, first_page, last_page, chunk_text, claims, company="", zoom=1.6):
    """
    Returns (sentences, [(page_number, png_bytes), ...]) for the pages that contain a highlight.
    If nothing could be matched, returns the first page without highlights.
    """
    sentences = supporting_sentences(chunk_text, claims, company)
    images = []
    with pymupdf.open(pdf_path) as doc:
        last_page = min(last_page, doc.page_count)
        for page_no in range(first_page, last_page + 1):
            page = doc[page_no - 1]
            words = page.get_text("words")
            hits = matching_word_boxes(words, sentences) if sentences else set()
            if not hits:
                continue
            for idx in hits:
                x0, y0, x1, y1 = words[idx][:4]
                page.draw_rect(pymupdf.Rect(x0 - 1, y0 - 1, x1 + 1, y1 + 1),
                               color=None, fill=(1, 0.9, 0), fill_opacity=0.4, overlay=True)
            images.append((page_no, page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png")))
        if not images and first_page <= doc.page_count:
            page = doc[first_page - 1]
            images.append((first_page, page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png")))
    return sentences, images