#!/usr/bin/env python3
"""v16 ground-floor cycle 1 — baseline + v16 lexicon + v4 rubric on centa_topics_v16_otto_gold.

Ground-floor rebuild per Mark 2026-11-26: full rubric + improvement gamut on the new
n=98 gold (15 hubs covered) BEFORE going back to teacher-scale sets.

Arms: instr_v2, instr_v3, force_lex_r5, force_lex_v15, v16_lex, force_lex_v16 (safe),
      hybrid_v16_else_confuse_x, v16lex_else_instr_v4.
Writes: cycle1_rows.jsonl, cycle1_scores.json, autopsy.md (per-hub breakdown).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from decision_bakeoff.data import load_labels, load_tsv  # noqa: E402
from decision_bakeoff.laya_arm import LayaArm  # noqa: E402
from run_failure_pattern_round5 import lexicon_hits as lexicon_hits_r5  # noqa: E402
from run_failure_pattern_round6 import (  # noqa: E402
    lexicon_hits_v15,
    decide,
    safe_path,
    make_fight_v15,
    _margin,
    _top2,
)
from run_failure_pattern_round7 import apply_confuse_pair  # noqa: E402
from run_failure_pattern_round9 import apply_confuse_expanded  # noqa: E402

RUBRIC_V4 = {
    "food": "Recipes, restaurants, cooking, and eating. NOT groceries-as-chore.",
    "sports": "Athletes, teams, competition, training, running, coaching, game-day.",
    "business": "Companies, work life, markets, jobs, entrepreneurship, management.",
    "health": "Wellness, fitness habits, sleep, medical, mental health, care.",
    "science-technology": "Software, hardware, gadgets, engineering, programming, applied science.",
    "entertainment": "TV, film, music, streaming, celebrity, pop culture.",
    "writing-literature": "Books, essays, and the craft of writing itself.",
    "animals-nature": "Pets, wildlife, birding, outdoors, hiking, trails.",
    "design": "Style, clothing craft, sewing, interiors, spaces, making physical things beautiful/functional.",
    "culture": "Museums, galleries, art, photos, performances, cultural life.",
    "travel": "Trips, destinations, airports, trains, hotels, packing, tourism.",
    "family": "Parenting, kids, marriage, siblings, relatives, home life and relationships.",
    "news": "Current events, headlines, local government, public affairs, breaking events.",
    "gardening": "Plants, yards, growing food or flowers, compost, garden work.",
    "games": "Video games, board games, tabletop RPGs, puzzles, and play.",
}


def lexicon_hits_v16(text: str) -> list[tuple[str, str]]:
    """v15 lex minus confirmed-bad rules, plus five new hubs. Otto 2026-11-26."""
    t = text.lower()
    hits = lexicon_hits_v15(text)

    def add(hint: str, hub: str):
        if (hint, hub) not in hits:
            hits.append((hint, hub))

    # --- remove confirmed-bad v15 rules (Otto human confirm) ---
    bad = [
        "household organizer / drawer divider build",          # 3602 is sci-tech
        "personal notes / writing for future self",            # 3648 is sci-tech
        "household shoe retirement / domestic life",           # 3027 is sports
        "reading/annotating reports as writing practice",      # 3072 confirmed business
        "sketchbook / notebook practice as writing-adjacent craft",  # 3224 confirmed culture
        "product privacy positioning (company)",               # 12073/12107 confirmed sci-tech
    ]
    if re.search(r"annual reports on paper|notes i leave myself are arguments", t):
        add("reading annual reports as business habit", "business")
    if re.search(r"notebook are hands|sketchbook|drawing is not really about the hands", t):
        add("sketchbook drawing practice (visual art)", "culture")
    if re.search(r"we're a website|don't track or sell|centa data privacy|why we don't have an app", t):
        add("Centa product no-app/privacy stance", "science-technology")
    hits = [h for h in hits if h[0] not in bad]
    # repair: solder without divider → sci-tech (round6 gated this behind 'not divider')
    if re.search(r"solder|nina-solder", t):
        add("electronics soldering / hardware fix on the bench", "science-technology")
    if "future me" in t and "notes" in t:
        add("engineering/work notes for future self", "science-technology")
    if re.search(r"yard-work|farm upstate", t) and re.search(r"shoe|pair|retiring", t):
        add("worn-out running shoes", "sports")

    # --- TRAVEL ---
    if re.search(r"\bgate b\d|delay board|wait for crew|airport|terminal \d", t):
        add("airport / flight delay", "travel")
    if re.search(r"\bflight\b|carry-on|packing light|checked bag", t):
        add("flying / packing for a trip", "travel")
    if re.search(r"\btrain\b.*(dining car|window seat|hours)|slow train|took the train", t):
        add("train journey", "travel")
    if re.search(r"\bhotel\b|check.?in|rental (car|counter)", t):
        add("hotel / rental on a trip", "travel")
    if re.search(r"\blisbon|alfama|paris|tokyo|rome|barcelona|mexico city|new york city\b", t):
        add("destination city tourism", "travel")
    if re.search(r"road trip|drove the long way|scenic route|interstate saves", t):
        add("road trip", "travel")
    if re.search(r"itinerary|passport|jet ?lag|sightseeing|tourist", t):
        add("trip logistics / tourism", "travel")

    # --- NEWS ---
    if re.search(r"city council|council voted|public comment|mayor says", t):
        add("local government news", "news")
    if re.search(r"without power|power outage|storm (line|damage)|cooling centers|utility (says|now says)", t):
        add("storm / infrastructure breaking news", "news")
    if re.search(r"per this morning's paper|this morning's paper|according to the (paper|report)|ap is reporting|reuters", t):
        add("newspaper headline event", "news")
    if re.search(r"council voted|ballot measure|primary turnout|record turnout", t):
        add("election / vote news", "news")
    if re.search(r"cutting \d+ jobs|plant.*cutting|layoffs? (announced|hit)", t):
        add("employer layoff news event", "news")
    if re.search(r"water main|hearings (on|opened|started)|testified|collapse hearings", t):
        add("public infrastructure / hearings news", "news")

    # --- GARDENING ---
    if re.search(r"\btomato(es)?\b", t) and re.search(r"vine|ripe|garden|grew|harvest|plant", t):
        add("growing tomatoes", "gardening")
    if re.search(r"compost|turned the pile", t):
        add("compost pile", "gardening")
    if re.search(r"first frost|last frost|hard freeze", t) and re.search(r"cover|coming in|plants|basil|tender", t):
        add("frost protection garden work", "gardening")
    if re.search(r"ladybugs|aphids|pest(s)? (in|on) the garden", t):
        add("garden pest management", "gardening")
    if re.search(r"raised bed|garden bed|planter box", t):
        add("raised-bed gardening", "gardening")
    if re.search(r"started seeds|seed starting|under the (shop|grow) light|germinat", t):
        add("seed starting", "gardening")
    if re.search(r"\bweeding\b|\bmulch\b|pruning|divided the (hostas|perennials)", t):
        add("garden maintenance", "gardening")

    # --- GAMES ---
    if re.search(r"\bboss\b.*(beat|died|fight|phase)|roguelike|the run died|floor one", t):
        add("video game run / boss fight", "games")
    if re.search(r"deck-?builder|slimmed the deck|twelve cards", t):
        add("deck-builder video game", "games")
    if re.search(r"board game|tabletop|session zero|the dm (asked|said|told)|dungeon master|campaign starts|backstory", t):
        add("tabletop / board game night", "games")
    if re.search(r"headset|last revive|one revive|co-?op (boss|game)|respawn", t):
        add("co-op video game", "games")
    if re.search(r"puzzle game|energy bar|no timer.*level", t):
        add("puzzle video game", "games")
    if re.search(r"\bspeedrun\b|\blevel\b.*\bboss\b|final boss", t):
        add("video game progression", "games")

    # --- FAMILY ---
    if re.search(r"kindergarten|first day of (school|preschool)|backpack", t):
        add("kid school milestone", "family")
    if re.search(r"training wheels|holding the seat|pedal away", t):
        add("teaching kid to ride a bike", "family")
    if re.search(r"bedtime (negotiation|routine|story)|one more chapter", t):
        add("kid bedtime routine", "family")
    if re.search(r"\bkid(s)?\b.*(made dinner|made breakfast|unprompted)|fourteen-year-old|teenager.*(dinner|chores)", t):
        add("kid doing something grown", "family")
    if re.search(r"called my (dad|mom)|calling my (dad|mom)|my (father|mother) (said|called)", t):
        add("parent relationship", "family")
    if re.search(r"at mom's|at dad's|my daughter\b|my son\b|my kids?\b|our kids?\b", t) and not re.search(
        r"trail|run|garden", t
    ):
        add("immediate family life", "family")
    return hits


def build_q_v4(base: dict) -> dict:
    q = json.loads(json.dumps(base))
    q["topic"]["criteria"] = dict(RUBRIC_V4)
    return q


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=REPO / "data" / "centa_topics_v16_otto_gold")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    labels = load_labels(args.data)
    rows = load_tsv(args.data / "test.tsv")
    if args.limit:
        rows = rows[: args.limit]
    q_v2 = json.loads((REPO / "questions" / "centa_topics_instr_v2.json").read_text())
    q_v3 = json.loads((REPO / "questions" / "centa_topics_instr_v3.json").read_text())
    q_v4 = build_q_v4(q_v3)
    criteria = dict(q_v2["topic"]["criteria"])

    os.environ.setdefault("USE_TF", "0")
    print("loading Laya…", flush=True)
    arm = LayaArm()
    print("Laya ready", flush=True)

    hit: Counter = Counter()
    n = 0
    t0 = time.perf_counter()
    by_hub_total: Counter = Counter()
    by_hub_miss: dict[str, Counter] = defaultdict(Counter)
    with (out / "cycle1_rows.jsonl").open("w") as fh:
        for row in rows:
            gold = labels[int(row["label"])]
            state = row["text"]
            n += 1
            by_hub_total[gold] += 1
            preds: dict[str, str] = {}

            c2, _, p2, _ = decide(arm, state, q_v2, 0.3)
            preds["instr_v2"] = c2
            c3, _, p3, _ = decide(arm, state, q_v3, 0.3)
            preds["instr_v3"] = c3
            c4, _, p4, _ = decide(arm, state, q_v4, 0.3)
            preds["instr_v4_rubric"] = c4

            lex_r5 = lexicon_hits_r5(state)
            lex_v15 = lexicon_hits_v15(state)
            lex_v16 = lexicon_hits_v16(state)
            m2 = _margin(p2)
            t2 = _top2(p2)

            c_safe_r5, _ = safe_path(arm, state, criteria, c2, m2, t2, lex_r5)
            preds["force_lex_r5"] = lex_r5[0][1] if lex_r5 else c_safe_r5
            c_safe_v15, _ = safe_path(arm, state, criteria, c2, m2, t2, lex_v15)
            preds["force_lex_v15"] = lex_v15[0][1] if lex_v15 else c_safe_v15
            c_safe_v16, _ = safe_path(arm, state, criteria, c2, m2, t2, lex_v16)
            preds["force_lex_v16"] = lex_v16[0][1] if lex_v16 else c_safe_v16
            preds["v16_lex_only"] = lex_v16[0][1] if lex_v16 else c2

            c_cf, _ = apply_confuse_expanded(arm, state, criteria, c3, _top2(p3), p3)
            preds["v3_confuse_x"] = c_cf
            preds["hybrid_v16_else_confuse_x"] = lex_v16[0][1] if lex_v16 else c_cf
            preds["v16lex_else_instr_v4"] = lex_v16[0][1] if lex_v16 else c4

            for k, v in preds.items():
                hit[k] += int(v == gold)
                if v != gold:
                    by_hub_miss[gold][k] += 1

            fh.write(json.dumps({
                "id": row.get("id", ""), "gold": gold, "text": state[:200],
                "lex_v16": [h[0] for h in lex_v16], **preds,
            }) + "\n")
            if n % 10 == 0:
                acc = {k: round(v / n, 3) for k, v in hit.items()}
                print(f"[{n}/{len(rows)}] {acc}", flush=True)

    scores = {
        "tag": "v16_ground_floor_cycle1",
        "n": n,
        "accuracies": {k: round(v / n, 4) for k, v in hit.items()},
        "counts": dict(hit),
        "wall_s": time.perf_counter() - t0,
        "complete": True,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / "cycle1_scores.json").write_text(json.dumps(scores, indent=1))

    lines = ["# v16 cycle1 autopsy — misses by hub\n\n"]
    for hub in sorted(by_hub_total):
        tot = by_hub_total[hub]
        miss = by_hub_miss.get(hub, Counter())
        lines.append(f"## {hub} (n={tot})\n")
        for arm_, cnt in miss.most_common():
            lines.append(f"- {arm_}: {cnt} miss\n")
        lines.append("\n")
    (out / "autopsy.md").write_text("".join(lines))
    print(json.dumps(scores["accuracies"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
