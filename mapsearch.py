#!/usr/bin/env python3
"""
mapsearch - find battlemaps by describing them ("forest at night", "ruined castle").

Uses a CLIP model that runs locally (on the Apple Silicon GPU when available).
Your image files are never moved, renamed or modified: the script only reads them
and keeps its own index in ~/.mapsearch/.

Setup (once, needs Python 3.9+):
    pip install sentence-transformers pillow numpy
The first `index` run downloads the CLIP model (about 600 MB); after that everything runs on your own
computer and nothing is uploaded anywhere. Edit MAPS_FOLDERS below to point at your maps.

Usage:
    python mapsearch.py index                           # build / update the index (uses MAPS_FOLDER below)
    python mapsearch.py index ~/Maps ~/OtherFolder      # or name the folder(s) here instead
    python mapsearch.py index --limit 100               # quick test on 100 images
    python mapsearch.py serve                           # search page at http://127.0.0.1:8765
    python mapsearch.py search "tavern interior at night" -n 10   # search from the terminal

Creators: the creator is the folder directly below the folder you index (index ~/Maps and the
folders ~/Maps/CreatorA, ~/Maps/CreatorB become creators). If your layout is deeper, for example
~/RPG/Maps/CreatorA, index ~/RPG and start with --creator-level 2, or index ~/RPG/Maps itself.
`python mapsearch.py creators` lists what was found. Indexes made before this feature need the
usual `index` command run once more (nothing is re-processed) to record the folders.

Types: every file is a map, a scene or a token, decided by your folder names (TYPE_RULES near the
top of this file, not by the model). `python mapsearch.py types` shows how many of each every
creator has under the current rules. Pick Maps, Scenes or Tokens in the page (or --kind on the
command line) to search only files of that type; this also changes how your words are phrased for
the model. Untick "Only this type" to ignore the labels. Changing TYPE_RULES needs a restart of
`serve`, not a re-index.

Ignored folders: KEEP_ONLY / IGNORE_FOLDERS (also near the top) leave whole folders out of the tool:
they are skipped when indexing and never appear in a search or in the counts. As shipped nothing is
ignored; the settings near the top show how.

Names: words in file and folder names are used too ("AbbeyPrison Courtyard" gives abbey, prison,
courtyard; "Dungeon - DragonGate/Dragon gate.jpg" also uses the folder). A file name counts most, its
folder less, higher folders little; the creator folder and words like Maps, Scenes, VTT, sizes and
numbers are ignored (TAG_NOISE), and words found in a large share of files count for almost nothing.
Names add to the image score and push down opposites (a file named Day when you ask for night); they
never hide a file. The grey-blue chips under each map are its name words (blue = matched your query);
click one to keep only files with that word. `python mapsearch.py tags` lists the most common words.
No re-index is needed; editing the lists needs a restart of `serve`.

Search tips: a leading dash after a comma excludes something ("forest, -ruins"); Loose / Normal /
Strict sets how closely a file must match; the % on each card is the match score.
"""
import argparse
import hashlib
import json
import math
import mimetypes
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None          # battlemaps are often huge
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- YOUR MAP FOLDER(S): put the folder(s) with your maps here, then just run `python mapsearch.py index` ---
# Example:  MAPS_FOLDERS = ["~/Documents/Maps"]     (Windows: ["C:/Users/you/Documents/Maps"])
# Folders given on the command line (`index <folder>`) are used instead of this list.
MAPS_FOLDERS = []

APP_DIR = Path(os.environ.get("MAPSEARCH_HOME", Path.home() / ".mapsearch"))
DB_PATH = APP_DIR / "index.sqlite"
THUMB_DIR = APP_DIR / "thumbs"

EXTS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_MODEL = "clip-ViT-B-32"
WORK_SIZE = 1792      # images are downscaled to this (longest side) before anything else
VIEW_SIZE = 448       # each view fed to CLIP is a letterboxed square of this size
TILE_MIN = 1000       # images at least this big (longest side, after downscale) also get tiles
THUMB_SIZE = 480
CHUNK = 16            # images per encoding step

# --- search tuning (query side only: changing these never requires re-indexing) ---
LOGIT_SCALE = 100.0   # CLIP's own softmax temperature
TILE_PENALTY = 0.8    # a match found only in one region counts a bit less than a whole-map match
SIMILAR_TEXT = 0.85   # background phrases this close to the query are not treated as competitors
STRICTNESS = {"loose": 0.05, "normal": 0.25, "strict": 0.6}   # minimum match score to be listed

# --- what counts as a map, a scene or a token: decided by your folder names (no model involved) ---
# Rules are checked in order and the first match wins; a file that matches none is TYPE_DEFAULT.
#   (type, "folder",  text)   any folder below the folder you indexed whose name contains the text
#   (type, "creator", name)   everything from that creator (exact folder name, any capitals); if the
#                             index predates the creator feature, any folder with that name counts
# Changing these takes effect the next time you start `serve`; nothing needs re-indexing.
TYPE_RULES = [
    ("tokens", "folder", "token"),
    ("scenes", "folder", "scene"),
    # ("scenes", "folder", "illustration"),   # example: add more rules like this, one per line
    # ("scenes", "creator", "SomeCreator"),   # example: everything from one creator is a scene
]
TYPE_DEFAULT = "maps"

# --- folders to leave out completely: never indexed, never shown in any search ---
# IGNORE_FOLDERS: any folder below the folder you index whose name contains one of these texts is left
#   out entirely (any capitals).
# KEEP_ONLY: for the folders named here (exact name), everything inside is left out EXCEPT the
#   sub-folders listed (exact names, at any depth). Example (remove the # to use it): from a creator's
#   folder called "Big Bundle" only its "Maps" and "VTT Tokens" folders are kept; its music, PDFs,
#   illustrations and so on are ignored. (A kept folder is then typed by TYPE_RULES as usual.)
# Files of an ignored folder that are already in the index disappear from searches when you restart
# `serve`, and are removed from the index the next time you run `index`. To bring a folder back,
# change the rules here and run `index` again (it will be read from scratch).
IGNORE_FOLDERS = []
KEEP_ONLY = {
    # "big bundle": ["maps", "vtt tokens"],
}

# A result must look more like your query than like these other kinds of place or creature.
PLACES = [
    "a forest", "a dense jungle", "grassy plains", "farmland", "a swamp", "a desert", "a snowy tundra",
    "mountains", "cliffs", "a river", "a lake", "a beach", "the ocean", "a ship deck", "a pirate ship",
    "harbor docks", "a cave", "an underground cavern", "a mine", "dungeon corridors", "a crypt", "a sewer",
    "a temple interior", "a castle exterior", "a castle courtyard", "a throne room", "a ruined castle",
    "fortress walls", "a tower", "a village", "a town street", "a city street", "a market square",
    "a tavern interior", "an inn", "a house interior", "a bedroom", "a shop", "a library", "a wizard tower",
    "a laboratory", "a prison", "an arena", "a graveyard", "a bridge", "a road", "a campsite", "a barn",
    "a mill", "a lighthouse", "a volcano", "lava", "an ice cave", "a frozen lake", "a space station",
    "a spaceship interior", "a sci-fi corridor", "a futuristic city", "an alien planet", "a wasteland",
    "ruins", "a ruined city", "a battlefield", "a military camp", "an airship", "a cathedral", "a monastery",
    "a palace", "a ballroom", "a garden", "a park", "a haunted house", "a mansion", "a warehouse",
    "a factory", "a canyon", "a waterfall", "an island", "a treasure vault", "a ritual circle",
    "a magic portal", "floating islands", "an underwater scene", "a tent", "an empty room", "a plain texture",
]
CREATURES = [
    "a goblin", "an orc", "a skeleton", "a zombie", "a dragon", "a wolf", "a bear", "a giant spider",
    "a knight in armor", "a wizard", "a rogue", "a cleric", "a bard", "an elf", "a dwarf", "a halfling",
    "a demon", "a devil", "an angel", "a giant", "an ogre", "a troll", "a slime", "a ghost", "a vampire",
    "a werewolf", "a mimic", "a floating eye monster", "a kraken", "a mermaid or merfolk", "a harpy",
    "a centaur", "a minotaur", "a golem", "a fire elemental", "an imp", "a giant rat", "a snake", "a bat",
    "a crow", "a horse", "a dog", "a cat", "a wild boar", "a merchant", "a town guard", "a noble",
    "a peasant", "a pirate", "an assassin", "a soldier", "a barbarian", "a paladin", "a druid", "a ranger",
    "a monk", "a necromancer", "a priest", "a cultist", "a bandit", "a treasure chest", "a barrel", "a door",
    "a trap", "a campfire", "a robot", "an alien", "a space marine", "a plant monster", "an insect swarm",
    "a lizardfolk", "a gnome", "a tiefling", "a beholder-like tyrant", "a skeletal dragon", "a statue",
]

# What "Maps / Scenes / Tokens / Anything" do: how the query is worded for the model (templates),
# and what it has to beat (background). Which files belong to a type is set by TYPE_RULES above.
KIND_ORDER = ["maps", "scenes", "tokens"]
KINDS = {
    "maps": {
        "templates": ["a top-down battle map of {}", "a fantasy tabletop RPG battlemap, {}"],
        "background": PLACES,
    },
    "scenes": {
        "templates": ["a fantasy illustration of {}", "concept art of {}, wide shot"],
        "background": PLACES,
    },
    "tokens": {
        "templates": ["a round character token of {}", "a tabletop RPG token illustration of {}"],
        "background": CREATURES,
    },
    "all": {
        "templates": ["a picture of {}", "{}"],
        "background": PLACES + CREATURES,
    },
}


def _unit(v):
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return (v / norm).astype(np.float32)


# --------------------------------------------------------------------------- database

def open_db():
    APP_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, check_same_thread=False)
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS files(
            id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, mtime REAL, size INTEGER, root TEXT);
        CREATE TABLE IF NOT EXISTS emb(
            file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
            view INTEGER NOT NULL, vec BLOB NOT NULL);
        CREATE INDEX IF NOT EXISTS emb_file ON emb(file_id);
        """
    )
    if "root" not in [r[1] for r in db.execute("PRAGMA table_info(files)")]:
        db.execute("ALTER TABLE files ADD COLUMN root TEXT")   # index made by an older version
        db.commit()
    return db


NO_CREATOR = "(no creator folder)"


def root_of(path, roots):
    """The outermost indexed folder that contains this file."""
    for r in sorted(roots, key=len):
        if path.startswith(r.rstrip(os.sep) + os.sep):
            return r
    return None


def creator_of(path, root, level=1):
    """Creator = the folder `level` steps below the folder you indexed ('' if the root is unknown)."""
    if not root:
        return ""
    parts = os.path.relpath(path, root).split(os.sep)
    return parts[level - 1] if len(parts) > level else NO_CREATOR


def type_of(path, root, creator):
    """'maps', 'scenes' or 'tokens' for a file, from TYPE_RULES (folder and creator names)."""
    rel = os.path.relpath(path, root) if root else path
    folders = [f.lower() for f in rel.split(os.sep)[:-1]]
    c = creator.lower()
    for kind, how, text in TYPE_RULES:
        text = text.lower()
        if how == "folder" and any(text in f for f in folders):
            return kind
        if how == "creator" and (c == text or (not root and text in folders)):
            return kind       # (without a recorded root, any folder with the creator's name counts)
    return TYPE_DEFAULT


def is_ignored(path, root):
    """True if the file is left out by IGNORE_FOLDERS or KEEP_ONLY (folders below the indexed one)."""
    rel = os.path.relpath(path, root) if root else path
    folders = [f.lower() for f in rel.split(os.sep)[:-1]]
    if any(t.lower() in f for t in IGNORE_FOLDERS for f in folders):
        return True
    for parent, keep in KEEP_ONLY.items():
        p = parent.lower()
        if p in folders:                              # the file is inside this folder...
            inside = folders[folders.index(p) + 1:]
            if not any(f in {k.lower() for k in keep} for f in inside):
                return True                           # ...but not inside one of the folders to keep
    return False


# --------------------------------------------------------------------------- name tags
# Words found in file and folder names ("AbbeyPrison Courtyard", "Roadside-campsite", a folder called
# "Dungeon - DragonGate") are used as extra evidence when you search. They are computed from the paths
# every time `serve` starts, so editing the lists below needs a restart of `serve`, never a re-index.
# Run `python mapsearch.py tags` to see which words are most common and decide what to add to TAG_NOISE.

NAME_WEIGHT = 0.35          # most a perfect name match can add to a file's 0-1 score (0 = ignore names)
OPPOSITE_PENALTY = 0.7      # a file named "Day" when you ask for "night" keeps this share of its score
EXCLUDE_PENALTY = 0.8       # a file named "Ruins" when you search "-ruins" loses up to this share
TAG_SOURCE_WEIGHTS = (1.0, 0.5, 0.2, 0.1)   # file name, its folder, then up to two folders above that
TAG_NOISE = {
    # words that say nothing about the picture (type folders, grid sizes, pack wording, filler)
    "map", "maps", "scene", "scenes", "token", "tokens", "vtt", "gridless", "grid", "gridded", "free",
    "mad", "pack", "bundle", "everything", "battlemap", "battlemaps", "battle", "version", "final",
    "copy", "print", "printable", "high", "low", "res", "hires", "lowres", "part", "pt", "the", "and",
    "for", "with", "from", "png", "jpg", "jpeg", "webp", "uhd", "foundry", "roll20", "dungeondraft",
    "variant", "alt", "alternate", "version", "numbered", "labeled", "unlabeled", "key",
}
TAG_SYNONYMS = {            # different words that should count as the same word
    "inside": "interior", "indoor": "interior", "indoors": "interior", "outside": "exterior",
    "outdoor": "exterior", "outdoors": "exterior", "woods": "forest", "nighttime": "night",
    "daytime": "day",
}
TAG_EMPHASIS = {"day": 1.5, "night": 1.5, "interior": 1.5, "exterior": 1.5}   # count extra in a query
TAG_OPPOSITE = {"day": "night", "night": "day", "interior": "exterior", "exterior": "interior"}

_SPLIT = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")


def _stem(w):
    w = TAG_SYNONYMS.get(w, w)
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 4 and w.endswith(("ches", "shes", "sses", "xes", "zes")):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    return TAG_SYNONYMS.get(w, w)


def tag_words(text):
    """'AbbeyPrison Courtyard [17x22]' -> [('abbey','abbey'), ('prison','prison'), ('courtyard','courtyard')].

    Each item is (stem, word as written). CamelCase, spaces, dashes, underscores and dots split words;
    numbers, sizes, one/two-letter bits and TAG_NOISE words are dropped.
    """
    text = re.sub(r"['’]s\b", "", text)
    out, seen = [], set()
    for chunk in re.split(r"[^A-Za-z0-9]+", text):
        for w in _SPLIT.split(chunk):
            lw = w.lower()
            if len(lw) < 3 or lw.isdigit() or lw in TAG_NOISE:
                continue
            s = _stem(lw)
            if s in TAG_NOISE or s in seen:
                continue
            seen.add(s)
            out.append((s, lw))
    return out


def path_tags(path, root, level=1):
    """Tags of one file: {stem: (weight, word)}, the stems that come from the file name, folder key."""
    if root:
        parts = os.path.relpath(path, root).split(os.sep)
        folders = parts[level:-1] if len(parts) > level else []   # the creator folder itself is skipped
    else:
        parts = path.split(os.sep)
        folders = parts[-4:-1]
    tags, from_file = {}, set()
    for rank, text in enumerate([Path(parts[-1]).stem] + list(reversed(folders))[:3]):
        for stem, word in tag_words(text):
            if stem not in tags:                  # nearest source wins, so it keeps the highest weight
                tags[stem] = (TAG_SOURCE_WEIGHTS[rank], word)
            if rank == 0:
                from_file.add(stem)
    return tags, from_file, os.sep.join(folders)


def split_query(text):
    """'forest at night, -ruins' -> ('forest at night', ['ruins'])."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    pos = ", ".join(p for p in parts if not p.startswith("-")) or text.strip("-, ")
    neg = [p[1:].strip() for p in parts if p.startswith("-") and p[1:].strip()]
    return pos, neg


def pick_model(db, requested):
    row = db.execute("SELECT value FROM meta WHERE key='model'").fetchone()
    if row:
        if requested and requested != row[0]:
            sys.exit(
                f"This index was built with {row[0]}. To switch to {requested}, "
                f"delete {DB_PATH} and index again."
            )
        return row[0]
    return requested or DEFAULT_MODEL


def load_model(name):
    import torch
    from sentence_transformers import SentenceTransformer

    if torch.cuda.is_available():
        device = "cuda"                    # NVIDIA GPU (Windows / Linux)
    elif torch.backends.mps.is_available():
        device = "mps"                     # Apple Silicon
    else:
        device = "cpu"
    print(f"Loading model {name} on {device} (first run downloads it)...")
    return SentenceTransformer(name, device=device)


# --------------------------------------------------------------------------- images

def thumb_name(path):
    return hashlib.sha1(path.encode("utf-8")).hexdigest() + ".jpg"


def load_rgb(path):
    """Open an image, flatten transparency onto grey, downscale to WORK_SIZE."""
    with Image.open(path) as im:
        if im.format == "JPEG":
            im.draft("RGB", (WORK_SIZE, WORK_SIZE))
        has_alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
        im = im.convert("RGBA" if has_alpha else "RGB")
        im.thumbnail((WORK_SIZE, WORK_SIZE), Image.Resampling.LANCZOS)
        if has_alpha:
            bg = Image.new("RGBA", im.size, (128, 128, 128, 255))
            bg.alpha_composite(im)
            im = bg.convert("RGB")
        return im


def letterbox(im, size=VIEW_SIZE):
    """Fit the whole image in a square (grey bars) so CLIP never crops the edges."""
    im = im.copy()
    im.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (128, 128, 128))
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return canvas


def save_thumb(im, path):
    t = im.copy()
    t.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.Resampling.LANCZOS)
    t.save(THUMB_DIR / thumb_name(path), "JPEG", quality=82)


def make_views(path, tiles=True):
    """View 0 is the whole map; the rest are the four quadrants and the centre."""
    im = load_rgb(path)
    save_thumb(im, path)
    views = [letterbox(im)]
    if tiles and max(im.size) >= TILE_MIN:
        w, h = im.size
        boxes = [
            (0, 0, w // 2, h // 2), (w // 2, 0, w, h // 2),
            (0, h // 2, w // 2, h), (w // 2, h // 2, w, h),
            (w // 4, h // 4, w - w // 4, h - h // 4),
        ]
        views += [letterbox(im.crop(b)) for b in boxes]
    return views


def iter_images(roots):
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")
                           and not any(t.lower() in d.lower() for t in IGNORE_FOLDERS)]
            for fn in filenames:
                if fn.startswith("."):
                    continue
                if os.path.splitext(fn)[1].lower() in EXTS:
                    yield os.path.join(dirpath, fn)


# --------------------------------------------------------------------------- indexing

def cmd_index(args):
    folders = args.folders or MAPS_FOLDERS
    if not folders:
        sys.exit("No map folder set. Either edit MAPS_FOLDERS near the top of this script, or run:\n"
                 "  python mapsearch.py index <your maps folder>")
    roots = [os.path.abspath(os.path.expanduser(r)) for r in folders]
    for r in roots:
        if not os.path.isdir(r):
            sys.exit(f"Not a folder (or drive not mounted): {r}")

    db = open_db()
    model_name = pick_model(db, args.model)

    found = sorted(set(iter_images(roots)))
    kept = [p for p in found if not is_ignored(p, root_of(p, roots))]
    if len(kept) != len(found):
        print(f"Skipping {len(found) - len(kept)} images in ignored folders (see KEEP_ONLY / IGNORE_FOLDERS).")
    found = kept
    print(f"Found {len(found)} images.")
    existing = {
        p: (fid, mtime, size, root)
        for fid, p, mtime, size, root in db.execute("SELECT id, path, mtime, size, root FROM files")
    }

    # forget files that were deleted or moved out of these folders
    found_set = set(found)
    prefixes = tuple(r.rstrip(os.sep) + os.sep for r in roots)
    gone = [(v[0],) for p, v in existing.items() if p.startswith(prefixes) and p not in found_set]
    if gone:
        db.executemany("DELETE FROM emb WHERE file_id=?", gone)
        db.executemany("DELETE FROM files WHERE id=?", gone)
        db.commit()
        print(f"Removed {len(gone)} files that no longer exist or are in ignored folders.")
        existing = {p: v for p, v in existing.items() if p in found_set or not p.startswith(prefixes)}

    # remember which indexed folder each file came from (the creator is the folder just below it);
    # this also fills it in for files indexed by an older version, without re-processing them
    relabel = [(root_of(p, roots), v[0]) for p, v in existing.items()
               if p in found_set and v[3] != root_of(p, roots)]
    if relabel:
        db.executemany("UPDATE files SET root=? WHERE id=?", relabel)
        db.commit()
        print(f"Recorded the folder of {len(relabel)} files (for the creator filter).")

    todo = []
    for p in found:
        try:
            st = os.stat(p)
        except OSError:
            continue
        old = existing.get(p)
        if old and abs(old[1] - st.st_mtime) < 1 and old[2] == st.st_size:
            continue
        todo.append((p, st.st_mtime, st.st_size))
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print("Nothing new to index.")
        return
    print(f"Indexing {len(todo)} images...")

    model = load_model(model_name)
    db.execute("INSERT OR REPLACE INTO meta VALUES('model', ?)", (model_name,))
    db.commit()

    tiles = not args.no_tiles

    def prepare(info):
        try:
            return info, make_views(info[0], tiles), None
        except Exception as e:  # unreadable / corrupt image
            return info, None, e

    chunks = [todo[i : i + CHUNK] for i in range(0, len(todo), CHUNK)]
    done = failed = 0
    start = time.time()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = [pool.submit(prepare, c) for c in chunks[0]]
            for ci in range(len(chunks)):
                current = pending
                if ci + 1 < len(chunks):  # decode the next chunk while this one is encoded
                    pending = [pool.submit(prepare, c) for c in chunks[ci + 1]]
                results = [f.result() for f in current]

                imgs, owner, ok = [], [], []
                for info, views, err in results:
                    if err is not None:
                        failed += 1
                        print(f"\n  skipped {info[0]}: {err}")
                        continue
                    for v in views:
                        imgs.append(v)
                        owner.append(len(ok))
                    ok.append(info)

                if imgs:
                    embs = model.encode(
                        imgs, batch_size=32, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False,
                    ).astype(np.float16)
                    owner = np.array(owner)
                    for k, (p, mtime, size) in enumerate(ok):
                        old = existing.get(p)
                        if old:
                            fid = old[0]
                            db.execute("DELETE FROM emb WHERE file_id=?", (fid,))
                            db.execute("UPDATE files SET mtime=?, size=?, root=? WHERE id=?",
                                       (mtime, size, root_of(p, roots), fid))
                        else:
                            fid = db.execute(
                                "INSERT INTO files(path, mtime, size, root) VALUES(?,?,?,?)",
                                (p, mtime, size, root_of(p, roots)),
                            ).lastrowid
                        db.executemany(
                            "INSERT INTO emb(file_id, view, vec) VALUES(?,?,?)",
                            [(fid, j, r.tobytes()) for j, r in enumerate(embs[owner == k])],
                        )
                    db.commit()

                done += len(current)
                rate = done / max(time.time() - start, 1e-6)
                eta = (len(todo) - done) / rate if rate else 0
                print(f"\r  {done}/{len(todo)}  {rate:.1f} img/s  ~{eta/60:.1f} min left   ",
                      end="", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is saved; run the same command again to continue.")
        return
    print(f"\nDone. {done - failed} indexed, {failed} skipped.")


# --------------------------------------------------------------------------- searching

class Index:
    """All embeddings in memory. A file has one vector per view; its score is the best view."""

    def __init__(self, creator_level=1):
        db = open_db()
        full = db.execute("SELECT id, path, root FROM files ORDER BY id").fetchall()
        rows = [(r[0], r[1]) for r in full]
        self.ids = np.array([r[0] for r in rows], dtype=np.int64)
        self.paths = [r[1] for r in rows]
        self.lpaths = [p.lower() for p in self.paths]
        self.creator = [creator_of(p, root, creator_level) for _, p, root in full]
        self.lcreator = [c.lower() for c in self.creator]
        self.types = [type_of(p, root, c) for (_, p, root), c in zip(full, self.creator)]
        self.type_arr = np.array(self.types)
        self.has_roots = any(root for _, _, root in full)
        self.ignored = np.array([is_ignored(p, root) for _, p, root in full], dtype=bool)  # see IGNORE_FOLDERS
        self.count = int((~self.ignored).sum())                                          # files that can be found
        counts = {}
        for c, skip in zip(self.creator, self.ignored):
            if c and not skip:
                counts[c] = counts.get(c, 0) + 1
        self.creators = sorted(counts.items(), key=lambda kv: kv[0].lower())
        self.pos = {fid: i for i, (fid, _) in enumerate(rows)}
        self._build_tags(full, creator_level)
        vecs, owner, views = [], [], []
        self.first = np.full(len(rows), -1, dtype=np.int64)  # row of each file's whole-map vector
        for fid, view, blob in db.execute("SELECT file_id, view, vec FROM emb ORDER BY file_id, view"):
            i = self.pos.get(fid)
            if i is None:
                continue
            if view == 0:
                self.first[i] = len(vecs)
            vecs.append(np.frombuffer(blob, dtype=np.float16))
            owner.append(i)
            views.append(view)
        self.M = np.vstack(vecs).astype(np.float32) if vecs else np.zeros((0, 1), np.float32)
        self.owner = np.array(owner, dtype=np.int64)
        self.view = np.array(views, dtype=np.int64)
        db.close()
        try:
            self.loaded_mtime = DB_PATH.stat().st_mtime
        except FileNotFoundError:
            self.loaded_mtime = None

    def _build_tags(self, full, level):
        """Name tags of every file, and how common each word is (common words count for little)."""
        self.tagsets, df, seen = [], {}, set()
        for i, (_, p, root) in enumerate(full):
            tags, from_file, dkey = path_tags(p, root, level)
            self.tagsets.append(tags)
            if self.ignored[i]:
                continue
            for s in tags:
                if s in from_file:
                    df[s] = df.get(s, 0) + 1
                elif (s, dkey) not in seen:       # a pack's folder name counts once, not once per file
                    seen.add((s, dkey))
                    df[s] = df.get(s, 0) + 1
        self.df = df
        n = max(self.count, 1)
        self.idf = {s: min(1.0, max(0.0, math.log(n / c) / 3.0)) for s, c in df.items()}
        lists = {}
        for i, tags in enumerate(self.tagsets):
            if self.ignored[i]:
                continue
            for s, (w, _) in tags.items():
                lists.setdefault(s, ([], []))
                lists[s][0].append(i)
                lists[s][1].append(w)
        self.inv = {s: (np.array(a, dtype=np.int64), np.array(b, dtype=np.float32))
                    for s, (a, b) in lists.items()}

    def tag_mask(self, tags):
        """True for files whose names contain every one of these word stems."""
        mask = np.ones(len(self.ids), dtype=bool)
        for s in tags:
            m = np.zeros(len(self.ids), dtype=bool)
            if s in self.inv:
                m[self.inv[s][0]] = True
            mask &= m
        return mask

    def name_adjust(self, best, pos, neg):
        """Blend name evidence into the scores. pos = [(stem, emphasis)], neg = [stem].

        A name match adds up to NAME_WEIGHT; a file named for the opposite (Day when you ask for night)
        or for an excluded word is pushed down. Names never hide a file, they only move it.
        """
        out = best.copy()
        pos = [(s, e) for s, e in pos if s in self.inv and self.idf.get(s, 0) >= 0.05]
        if pos:
            num = np.zeros(len(out), dtype=np.float32)
            den = 0.0
            asked = {s for s, _ in pos}
            for s, emph in pos:
                idx, w = self.inv[s]
                num[idx] += emph * w * self.idf[s]
                den += emph
                opp = TAG_OPPOSITE.get(s)
                if opp in self.inv and opp not in asked:
                    have = np.zeros(len(out), dtype=bool)
                    have[idx[w >= 0.5]] = True
                    oi, ow = self.inv[opp]
                    bad = oi[ow >= 0.5]
                    out[bad[~have[bad]]] *= OPPOSITE_PENALTY
            out += (NAME_WEIGHT * num / den).astype(np.float32)
        for s in neg:
            if s in self.inv:
                idx, w = self.inv[s]
                out[idx] *= (1.0 - EXCLUDE_PENALTY * w)
        return out

    def browse(self, n, folder="", creator="", ok=None, tags=()):
        """Files matching the filters only (no query), in path order."""
        mask = ~self.ignored
        fm = self._folder_mask(folder, creator, tags)
        if fm is not None:
            mask = mask & fm
        if ok is not None:
            mask = mask & ok
        idx = sorted(np.flatnonzero(mask).tolist(), key=lambda i: self.lpaths[i])[:n]
        return [(int(i), None) for i in idx]

    def _folder_mask(self, folder, creator="", tags=()):
        """Files whose path contains `folder`, whose creator folder is exactly `creator`, and whose names
        contain every word in `tags`."""
        mask = None
        if folder:
            f = folder.lower()
            mask = np.fromiter((f in p for p in self.lpaths), dtype=bool, count=len(self.lpaths))
        if creator:
            c = creator.lower()
            cm = np.fromiter((x == c for x in self.lcreator), dtype=bool, count=len(self.lcreator))
            mask = cm if mask is None else (mask & cm)
        if tags:
            tm = self.tag_mask(tags)
            mask = tm if mask is None else (mask & tm)
        return mask

    def _top(self, scores, n, folder, min_score=-np.inf, creator="", tags=()):
        """Best n files as (position, score), skipping filtered-out files and scores below min_score."""
        scores = scores.copy()
        fm = self._folder_mask(folder, creator, tags)
        if fm is not None:
            scores[~fm] = -np.inf
        order = np.argsort(-scores)[:n]
        return [(int(i), float(scores[i])) for i in order
                if np.isfinite(scores[i]) and scores[i] >= min_score]

    def score(self, q, bg):
        """Per-file score = how strongly the file matches the query *compared with* the competitors.

        For every stored view, a softmax is taken over [query, competing phrases]; a greenish map
        that fits "swamp" or "grassy plains" better than "forest" scores low. A file's score is
        its best view. Returns an array with one score per file (-inf when a file has no views).
        """
        best = np.full(len(self.ids), -np.inf, dtype=np.float32)
        if not len(self.M):
            return best
        targets = np.vstack([q[None, :], bg]).astype(np.float32)
        sims = self.M @ targets.T                          # views x (1 + competitors)
        z = LOGIT_SCALE * sims
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        p = e[:, 0] / e.sum(axis=1)
        p[self.view > 0] *= TILE_PENALTY
        score = (p + 1e-3 * sims[:, 0]).astype(np.float32)  # raw similarity only breaks ties
        np.maximum.at(best, self.owner, score)
        best[self.ignored] = -np.inf
        return best

    def kind_ok(self, kind):
        """True for files whose folder-based type (see TYPE_RULES) is `kind`."""
        return self.type_arr == kind

    def pick(self, best, n, folder="", min_score=-np.inf, ok=None, creator="", tags=()):
        """Top n files as (position, score), plus how many matches the type filter hid."""
        scores = best.copy()
        fm = self._folder_mask(folder, creator, tags)
        if fm is not None:
            scores[~fm] = -np.inf
        hidden = 0
        if ok is not None:
            hidden = int(((scores >= min_score) & np.isfinite(scores) & ~ok).sum())
            scores[~ok] = -np.inf
        order = np.argsort(-scores)[:n]
        hits = [(int(i), float(scores[i])) for i in order
                if np.isfinite(scores[i]) and scores[i] >= min_score]
        return hits, hidden

    def similar(self, fid, n=60, folder="", creator="", tags=()):
        i = self.pos.get(fid)
        if i is None or self.first[i] < 0:
            return []
        q = self.M[self.first[i]]
        valid = self.first >= 0
        scores = self.M[np.clip(self.first, 0, None)] @ q
        scores[~valid | self.ignored] = -np.inf
        return self._top(scores, n, folder, creator=creator, tags=tags)

    def chips(self, pos, hits=()):
        """Name words shown under a card: the ones that matched your query first, then the file's own."""
        out = []
        for s, (w, word) in self.tagsets[pos].items():
            hit = s in hits
            if hit or (w >= 0.5 and self.idf.get(s, 0) >= 0.15):
                out.append({"k": s, "t": word, "hit": hit})
        out.sort(key=lambda c: not c["hit"])
        return out[:8]

    def describe(self, pos, score=None, hits=()):
        parts = Path(self.paths[pos]).parts
        return {"id": int(self.ids[pos]), "name": parts[-1], "dir": "/".join(parts[-3:-1]),
                "creator": self.creator[pos], "type": self.types[pos], "tags": self.chips(pos, hits),
                "score": None if score is None else round(min(1.0, max(0.0, score)), 2)}


class App:
    def __init__(self, model, creator_level=1):
        self.model = model
        self.model_lock = threading.Lock()
        self.creator_level = creator_level
        self.index = Index(creator_level)
        self._bg = {}       # competing-phrase embeddings per search type, built on first use

    def fresh(self):
        try:
            if DB_PATH.stat().st_mtime != self.index.loaded_mtime:
                self.index = Index(self.creator_level)  # picks up newly indexed files without a restart
        except FileNotFoundError:
            pass
        return self.index

    def embed(self, texts):
        with self.model_lock:
            return self.model.encode(texts, batch_size=64, convert_to_numpy=True,
                                     normalize_embeddings=True, show_progress_bar=False)

    def encode_phrases(self, phrases, templates):
        """Embed each phrase inside several sentence templates and average (CLIP likes sentences)."""
        texts = [t.replace("{}", p) for p in phrases for t in templates]
        v = self.embed(texts).reshape(len(phrases), len(templates), -1).mean(axis=1)
        return _unit(v)

    def background(self, kind):
        if kind not in self._bg:
            k = KINDS[kind]
            self._bg[kind] = self.encode_phrases(k["background"], k["templates"])
        return self._bg[kind]

    def query_vectors(self, text, kind="maps"):
        """'forest at night, -ruins' -> query vector for the first part, 'ruins' as an extra competitor."""
        k = KINDS[kind]
        pos, neg = split_query(text)
        q = self.encode_phrases([pos], k["templates"])[0]
        bg_all = self.background(kind)
        bg = bg_all[(bg_all @ q) <= SIMILAR_TEXT]   # drop near-synonyms of the query
        if neg:
            bg = np.vstack([bg, self.encode_phrases(neg, k["templates"])])
        return q, bg

    @staticmethod
    def words(text):
        """Name words of a query: ([(stem, emphasis)] wanted, [stem] excluded)."""
        pos, neg = split_query(text)
        return ([(s, TAG_EMPHASIS.get(s, 1.0)) for s, _ in tag_words(pos)],
                [s for s, _ in tag_words(" ".join(neg))])

    def text_search(self, index, text, n, folder="", min_score=STRICTNESS["normal"],
                    kind="maps", only=True, creator="", tags=(), names=True):
        """Returns (hits, weak, hidden).

        kind picks the wording and competitors (maps / scenes / tokens / all). With only=True,
        files whose folder-based type is another one are left out; hidden counts the matches
        that removed. If nothing clears min_score, the closest few come back with weak=True.
        tags = word stems every result's name must contain; names=True lets file and folder names
        add to (or take away from) the image score.
        """
        if kind not in KINDS:
            kind = "maps"
        q, bg = self.query_vectors(text, kind)
        best = index.score(q, bg)
        if names:
            best = index.name_adjust(best, *self.words(text))
        ok = index.kind_ok(kind) if only and kind in KIND_ORDER else None
        hits, hidden = index.pick(best, n, folder, min_score, ok, creator, tags)
        if hits:
            return hits, False, hidden
        hits, _ = index.pick(best, min(n, 12), folder, -np.inf, ok, creator, tags)
        return hits, True, hidden


PAGE = r"""<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Map search</title>
<style>
 body{font-family:-apple-system,system-ui,sans-serif;margin:0;background:#16161a;color:#e8e8ec}
 header{position:sticky;top:0;background:#1f1f25;padding:12px 16px;display:flex;gap:8px;flex-wrap:wrap;align-items:center;z-index:2}
 input,select,button{font:inherit;padding:8px 10px;border-radius:6px;border:1px solid #3a3a44;background:#2a2a32;color:inherit}
 button{cursor:pointer}
 #q{flex:1;min-width:260px}
 #status{padding:12px 16px 0;opacity:.7;font-size:14px}
 #grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px;padding:16px}
 .card{background:#1f1f25;border-radius:8px;overflow:hidden}
 .card img{width:100%;aspect-ratio:4/3;object-fit:contain;background:#000;display:block}
 .meta{padding:8px 10px;font-size:12px;line-height:1.4;word-break:break-all}
 .name{font-weight:600}.dir{opacity:.6}
 .actions{display:flex;gap:12px;margin-top:6px}
 .link{color:#8ab4ff;cursor:pointer}
 .chip{display:inline-block;margin:5px 4px 0 0;padding:1px 8px;border-radius:10px;background:#2a2a32;cursor:pointer;font-size:11px}
 .chip:hover{background:#3a3a44}.chip.hit{background:#3d5a99}
</style>
<header>
 <select id="kind" title="What are you looking for?"><option value="maps" selected>Maps</option><option value="scenes">Scenes</option><option value="tokens">Tokens</option></select>
 <select id="creator" title="Only files from this creator's folder" style="display:none"><option value="">All creators</option></select>
 <input id="q" placeholder="forest at night, ruined castle, -dungeon  (a leading dash excludes)" autofocus>
 <input id="folder" placeholder="only paths containing... (e.g. a creator or pack name)" size="28">
 <input id="tags" placeholder="name words (click a tag below a map)" size="22" title="Only files whose file or folder names contain all of these words">
 <label title="Let words in file and folder names (Night, Courtyard...) help the ranking"><input type="checkbox" id="names" checked> Use names</label>
 <select id="match" title="How closely a map must match"><option value="loose">Loose</option><option value="normal" selected>Normal</option><option value="strict">Strict</option></select>
 <label title="Only files your folders label as this type (see TYPE_RULES in the script). Untick to search everything."><input type="checkbox" id="only" checked> Only this type</label>
 <select id="n"><option>30</option><option selected>60</option><option>120</option><option>240</option></select>
 <button onclick="go()">Search</button>
</header>
<div id="status"></div>
<div id="grid"></div>
<script>
const $=id=>document.getElementById(id);
const esc=s=>s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let lastTotal=-1;
async function loadCreators(){
  try{
    const d=await (await fetch('/api/creators')).json();
    const sel=$('creator'), cur=sel.value;
    sel.innerHTML='<option value="">All creators</option>'+d.creators.map(c=>
      `<option value="${esc(c.name)}">${esc(c.name)} (${c.count})</option>`).join('');
    sel.value=cur;
    sel.style.display=(d.has_roots&&d.creators.length)?'':'none';
  }catch(e){}
}
async function run(url){
  $('status').textContent='Searching...';
  const d=await (await fetch(url)).json();
  if(d.total!==lastTotal){lastTotal=d.total;loadCreators();}  // index changed: refresh the creator list
  const found=(d.results||[]).length;
  let msg=d.error||(d.weak?(found?'No strong matches. Showing the '+found+' closest (try Loose, or different words).'
     :'Nothing found among files labelled as this type. Try another type, or untick "Only this type".')
     :found+' results ('+d.total+' images indexed)');
  if(d.hidden) msg+=' - '+d.hidden+' more matched but are labelled as another type (untick "Only this type" to see them)';
  $('status').textContent=msg;
  $('grid').innerHTML=(d.results||[]).map(x=>
   `<div class="card"><a href="/file?id=${x.id}" target="_blank"><img loading="lazy" src="/thumb?id=${x.id}"></a>
    <div class="meta"><div class="name">${esc(x.name)}${x.score!=null?' <span style="opacity:.55;font-weight:400">'+Math.round(x.score*100)+'%</span>':''}</div><div class="dir">${esc(x.dir)}</div>
    <div>${(x.tags||[]).map(c=>`<span class="chip${c.hit?' hit':''}" onclick="addTag('${c.k}')">${esc(c.t)}</span>`).join('')}</div>
    <div class="actions"><span class="link" onclick="reveal(${x.id})">Show in folder</span>
    <span class="link" onclick="similar(${x.id})">Similar</span></div></div></div>`).join('');
  window.scrollTo(0,0);
}
function params(extra){return new URLSearchParams({...extra,folder:$('folder').value.trim(),n:$('n').value,match:$('match').value,kind:$('kind').value,only:$('only').checked?'1':'0',creator:$('creator').value,tags:$('tags').value.trim(),names:$('names').checked?'1':'0'});}
function go(){const q=$('q').value.trim();if(q||$('tags').value.trim())run('/api/search?'+params({q}));}
function addTag(k){const w=$('tags').value.split(/[ ,]+/).filter(Boolean);if(!w.includes(k))w.push(k);$('tags').value=w.join(' ');go();}
function similar(id){run('/api/similar?'+params({id}));}
async function reveal(id){await fetch('/api/reveal?id='+id,{method:'POST'});}
$('q').addEventListener('keydown',e=>{if(e.key==='Enter')go();});
$('folder').addEventListener('keydown',e=>{if(e.key==='Enter')go();});
$('tags').addEventListener('keydown',e=>{if(e.key==='Enter')go();});
for(const id of ['creator','kind','match','only','names']) $(id).addEventListener('change',go);  // re-run when a filter changes
loadCreators();
</script>
"""


class Handler(BaseHTTPRequestHandler):
    app = None

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="text/html; charset=utf-8", cache=False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", "max-age=86400")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _lookup(self, index, qs):
        try:
            fid = int(qs.get("id", ["-1"])[0])
        except ValueError:
            return None
        pos = index.pos.get(fid)
        return None if pos is None else index.paths[pos]

    def do_GET(self):
        try:
            u = urlparse(self.path)
            qs = parse_qs(u.query)
            app = self.app
            index = app.fresh()
            n = max(1, min(300, int(qs.get("n", ["60"])[0])))
            folder = qs.get("folder", [""])[0]
            creator = qs.get("creator", [""])[0]

            if u.path == "/":
                return self._send(200, PAGE.encode())
            if u.path == "/api/creators":
                return self._json({"creators": [{"name": c, "count": k} for c, k in index.creators],
                                   "has_roots": index.has_roots, "level": app.creator_level})
            if u.path == "/api/search":
                q = qs.get("q", [""])[0].strip()
                thr = STRICTNESS.get(qs.get("match", ["normal"])[0], STRICTNESS["normal"])
                kind = qs.get("kind", ["maps"])[0]
                only = qs.get("only", ["1"])[0] == "1"
                tags = [s for s, _ in tag_words(qs.get("tags", [""])[0])]
                names = qs.get("names", ["1"])[0] == "1"
                if q:
                    hits, weak, hidden = app.text_search(index, q, n, folder, thr, kind, only, creator,
                                                         tags, names)
                elif tags:                    # only name words: list the files that have them
                    ok = index.kind_ok(kind) if only and kind in KIND_ORDER else None
                    hits, weak, hidden = index.browse(n, folder, creator, ok, tags), False, 0
                else:
                    hits, weak, hidden = [], False, 0
                shown = {s for s, _ in app.words(q)[0]} | set(tags)
                return self._json({"results": [index.describe(i, s, shown) for i, s in hits],
                                   "weak": weak, "hidden": hidden, "total": index.count})
            if u.path == "/api/similar":
                try:
                    fid = int(qs.get("id", ["-1"])[0])
                except ValueError:
                    fid = -1
                hits = index.similar(fid, n, folder, creator,
                                     [s for s, _ in tag_words(qs.get("tags", [""])[0])])
                return self._json({"results": [index.describe(i) for i, _ in hits],
                                   "weak": False, "hidden": 0, "total": index.count})
            if u.path == "/thumb":
                path = self._lookup(index, qs)
                if not path:
                    return self._send(404, b"not found", "text/plain")
                tp = THUMB_DIR / thumb_name(path)
                if not tp.exists():
                    save_thumb(load_rgb(path), path)
                return self._send(200, tp.read_bytes(), "image/jpeg", cache=True)
            if u.path == "/file":
                path = self._lookup(index, qs)
                if not path or not os.path.exists(path):
                    return self._send(404, b"not found", "text/plain")
                ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
                return self._send(200, Path(path).read_bytes(), ctype)
            return self._send(404, b"not found", "text/plain")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            try:
                self._json({"error": f"{type(e).__name__}: {e}", "results": [], "total": 0}, 500)
            except Exception:
                pass

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/reveal":
            path = self._lookup(self.app.fresh(), parse_qs(u.query))
            if path and os.path.exists(path):
                if sys.platform == "darwin":
                    subprocess.run(["open", "-R", path], check=False)                 # Finder
                elif os.name == "nt":
                    subprocess.run(["explorer", "/select,", os.path.normpath(path)], check=False)  # Explorer
                else:
                    subprocess.run(["xdg-open", os.path.dirname(path)], check=False)  # file manager
                return self._json({"ok": True})
        self._send(404, b"not found", "text/plain")


NEEDS_ROOTS = ("The creator filter needs one more pass: run your usual `index` command again. "
               "Nothing is re-processed; it only records which folder each file came from.")


def start_app(creator_level=1):
    db = open_db()
    n = db.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    if n == 0:
        sys.exit("The index is empty. Run first:  python mapsearch.py index <your maps folder>")
    name = pick_model(db, None)
    db.close()
    return App(load_model(name), creator_level)


def cmd_serve(args):
    app = start_app(args.creator_level)
    Handler.app = app
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)  # local machine only
    url = f"http://127.0.0.1:{args.port}"
    skipped = len(app.index.ids) - app.index.count
    print(f"{app.index.count} images indexed"
          + (f" ({skipped} more are in ignored folders and hidden)" if skipped else "")
          + f". Search page: {url}  (Ctrl-C to stop)")
    if not app.index.has_roots:
        print(NEEDS_ROOTS)
    elif app.index.creators:
        print(f"{len(app.index.creators)} creators found (the folder {args.creator_level} level(s) "
              f"below the folders you indexed).")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


def cmd_creators(args):
    """List the creator folders found, with file counts (a quick check that the level is right)."""
    db = open_db()
    index = Index(args.creator_level)
    db.close()
    if not len(index.ids):
        sys.exit("The index is empty.")
    if not index.has_roots:
        sys.exit(NEEDS_ROOTS)
    for name, count in index.creators:
        print(f"{count:6d}  {name}")


def cmd_types(args):
    """Show how many maps / scenes / tokens each creator has under the current TYPE_RULES."""
    index = Index(args.creator_level)
    if not len(index.ids):
        sys.exit("The index is empty.")
    if not index.has_roots:
        print(NEEDS_ROOTS + "\n")
    kinds = KIND_ORDER + [t for t in sorted(set(index.types)) if t not in KIND_ORDER]
    table = {}
    for creator, t, skip in zip(index.creator, index.types, index.ignored):
        if skip:
            continue                      # ignored files (KEEP_ONLY / IGNORE_FOLDERS) are not counted
        row = table.setdefault(creator or "(unknown folder)", dict.fromkeys(kinds, 0))
        row[t] += 1
    if not table:
        sys.exit("Everything in the index is in ignored folders (see KEEP_ONLY / IGNORE_FOLDERS).")
    width = max(len(c) for c in table) + 2
    print(f"{'creator':<{width}}" + "".join(f"{k:>9}" for k in kinds))
    for creator in sorted(table, key=str.lower):
        print(f"{creator:<{width}}" + "".join(f"{table[creator][k]:>9}" for k in kinds))
    print(f"{'total':<{width}}" + "".join(f"{sum(r[k] for r in table.values()):>9}" for k in kinds))


def cmd_tags(args):
    """List the most common name words, to decide what belongs in TAG_NOISE."""
    index = Index(args.creator_level)
    if not index.count:
        sys.exit("The index is empty.")
    words = {}
    for tags in index.tagsets:
        for s, (_, word) in tags.items():
            words.setdefault(s, word)
    top = sorted(index.df.items(), key=lambda kv: -kv[1])[:args.n]
    print(f"{len(index.df)} different words in the names of {index.count} files. Most common first;\n"
          f"'weight' is how much a match on that word counts (0 = ignored, 1 = full). Words that are\n"
          f"common but say nothing about a map belong in TAG_NOISE near the top of the script.\n")
    print(f"{'files':>7}  {'weight':>6}  word")
    for s, c in top:
        print(f"{c:7d}  {index.idf[s]:6.2f}  {words.get(s, s)}")


def cmd_search(args):
    app = start_app(args.creator_level)
    index = app.index
    if args.creator and not index.has_roots:
        sys.exit(NEEDS_ROOTS)
    hits, weak, hidden = app.text_search(index, args.query, args.n, args.folder or "",
                                         STRICTNESS[args.match], args.kind, not args.any_type,
                                         args.creator or "", [s for s, _ in tag_words(getattr(args, "tags", None) or "")],
                                         not getattr(args, "no_names", False))
    if hidden:
        print(f"({hidden} matches are labelled as other types; use --any-type to see them)", file=sys.stderr)
    if weak:
        print("No strong matches; closest few:", file=sys.stderr)
    for i, s in hits:
        print(f"{s:.2f}\t{index.paths[i]}")


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Search battlemaps by describing them.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="build or update the index (safe to re-run)")
    p.add_argument("folders", nargs="*", help="folders to scan recursively (default: MAPS_FOLDERS in the script)")
    p.add_argument("--model", help=f"CLIP model (default {DEFAULT_MODEL}; clip-ViT-L-14 is better but slower)")
    p.add_argument("--no-tiles", action="store_true", help="faster: skip the per-region views of large maps")
    p.add_argument("--limit", type=int, help="only index this many new images (for a test run)")
    p.add_argument("--workers", type=int, default=4, help="image decoding threads")
    p.set_defaults(fn=cmd_index)

    p = sub.add_parser("serve", help="start the local search page")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-open", action="store_true", help="do not open the browser")
    p.add_argument("--creator-level", type=int, default=1, metavar="N",
                   help="which folder below the indexed folder is the creator (default 1)")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("types", help="show how many maps, scenes and tokens each creator has")
    p.add_argument("--creator-level", type=int, default=1, metavar="N",
                   help="which folder below the indexed folder is the creator (default 1)")
    p.set_defaults(fn=cmd_types)

    p = sub.add_parser("creators", help="list the creator folders found and how many files each has")
    p.add_argument("--creator-level", type=int, default=1, metavar="N",
                   help="which folder below the indexed folder is the creator (default 1)")
    p.set_defaults(fn=cmd_creators)

    p = sub.add_parser("search", help="search from the terminal")
    p.add_argument("query")
    p.add_argument("-n", type=int, default=20)
    p.add_argument("--folder", help="only paths containing this text")
    p.add_argument("--match", choices=list(STRICTNESS), default="normal", help="how strict (default normal)")
    p.add_argument("--kind", choices=list(KINDS), default="maps", help="what you are looking for (default maps)")
    p.add_argument("--any-type", action="store_true", help="also show files labelled as another type")
    p.add_argument("--creator", help="only files from this creator's folder (see the `creators` command)")
    p.add_argument("--tags", help="only files whose names contain all these words, e.g. 'night bridge'")
    p.add_argument("--no-names", action="store_true", help="rank by the image only, ignoring file names")
    p.add_argument("--creator-level", type=int, default=1, metavar="N",
                   help="which folder below the indexed folder is the creator (default 1)")
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("tags", help="list the most common words in file and folder names")
    p.add_argument("-n", type=int, default=80, help="how many words to show (default 80)")
    p.add_argument("--creator-level", type=int, default=1, metavar="N",
                   help="which folder below the indexed folder is the creator (default 1)")
    p.set_defaults(fn=cmd_tags)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
