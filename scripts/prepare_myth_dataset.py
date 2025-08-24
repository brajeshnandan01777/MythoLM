
import argparse, os, re
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

DEFAULT_SOURCES = {
    "iliad": "https://www.gutenberg.org/files/2199/2199-h/2199-h.htm",
    "odyssey": "https://www.gutenberg.org/files/1727/1727-h/1727-h.htm",
    "mabinogion": "https://www.gutenberg.org/files/5160/5160-h/5160-h.htm",
    "kalevala": "https://www.gutenberg.org/files/5186/5186-h/5186-h.htm",
    "ramayana_txt": "https://www.gutenberg.org/ebooks/24869.txt.utf-8",
    "poetic_edda_index": "https://www.sacred-texts.com/neu/poe/index.htm",
    "prose_edda_index": "https://www.sacred-texts.com/neu/pre/index.htm",
    "mahabharata_index": "https://www.sacred-texts.com/hin/maha/index.htm",
}

HEADERS = {"User-Agent": "MythoLM-pureAR/1.0"}

def fetch(url):
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.text

def html_to_text(html, source_hint=""):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script","style","header","footer","nav"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    if "Gutenberg" in source_hint or "*** START OF THE PROJECT GUTENBERG EBOOK" in text:
        start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
        end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
        if start_marker in text and end_marker in text:
            text = text.split(start_marker,1)[-1].split(end_marker,1)[0]
    text = re.sub(r'\r', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def get_links(index_url, pattern):
    html = fetch(index_url)
    soup = BeautifulSoup(html, "html.parser")
    out = []
    import re as _re
    for a in soup.find_all("a", href=True):
        if _re.search(pattern, a["href"]):
            out.append(urljoin(index_url, a["href"]))
    seen = set(); ordered = []
    for u in out:
        if u not in seen:
            seen.add(u); ordered.append(u)
    return ordered

def main():
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--include", nargs="*", default=["iliad","odyssey","poetic_edda","prose_edda","mahabharata","ramayana","mabinogion","kalevala"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pieces = []

    if "iliad" in args.include:
        pieces.append(("Iliad", html_to_text(fetch(DEFAULT_SOURCES["iliad"]), "Gutenberg")))
    if "odyssey" in args.include:
        pieces.append(("Odyssey", html_to_text(fetch(DEFAULT_SOURCES["odyssey"]), "Gutenberg")))
    if "mabinogion" in args.include:
        pieces.append(("Mabinogion", html_to_text(fetch(DEFAULT_SOURCES["mabinogion"]), "Gutenberg")))
    if "kalevala" in args.include:
        pieces.append(("Kalevala", html_to_text(fetch(DEFAULT_SOURCES["kalevala"]), "Gutenberg")))
    if "ramayana" in args.include:
        txt = fetch(DEFAULT_SOURCES["ramayana_txt"])
        pieces.append(("Ramayana", html_to_text(txt, "Gutenberg")))
    if "poetic_edda" in args.include:
        for u in get_links(DEFAULT_SOURCES["poetic_edda_index"], r'poe\d+\.htm$'):
            pieces.append((f"Poetic Edda {u}", html_to_text(fetch(u), "sacred-texts")))
    if "prose_edda" in args.include:
        for u in get_links(DEFAULT_SOURCES["prose_edda_index"], r'pre\d+\.htm$'):
            pieces.append((f"Prose Edda {u}", html_to_text(fetch(u), "sacred-texts")))
    if "mahabharata" in args.include:
        for u in get_links(DEFAULT_SOURCES["mahabharata_index"], r'mb\d+\.htm$'):
            pieces.append((f"Mahabharata {u}", html_to_text(fetch(u), "sacred-texts")))

    sep = "\n\n############ MYTHOLM_SOURCE_BREAK ############\n\n"
    corpus = sep.join([f"### {t}\n\n{s}" for t,s in pieces])
    open(os.path.join(args.out_dir,"myth_corpus.txt"),"w",encoding="utf-8").write(corpus)

    cut = int(len(corpus)* (1-args.val_ratio))
    open(os.path.join(args.out_dir,"train.txt"),"w",encoding="utf-8").write(corpus[:cut])
    open(os.path.join(args.out_dir,"val.txt"),"w",encoding="utf-8").write(corpus[cut:])
    print("Saved data to", args.out_dir)

if __name__ == "__main__":
    main()
