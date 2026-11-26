#!/usr/bin/env python3
"""v16.1 lex — post-filter v16 hits with strict wild-text gates.

n24 autopsy: travel lex fired 53x @ 25% precision (city names, bare flight/train/hotel
catch Olympics/venue/economy news). news 2/8, family 2/4, games 0/3.
v16.1 keeps v16 hits but travel/news/family/games hints must ALSO match strict
first-person / event-frame patterns. Other hubs pass through unchanged.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from run_v16_cycle1 import lexicon_hits_v16  # noqa: E402

STRICT_TRAVEL = re.compile(
    r"gate\s+[a-z]?\d|delay board|wait for crew|carry-?on|packing (light|for)|"
    r"took the (slow )?train|dining car|drove the long way|checked (in|into)|"
    r"hotel (gave|room|lobby)|rental counter|jet\s?-?lag|itinerary|"
    r"(my|our|we)\s+(flight|road trip|trip)|my flight|flight (was |got )?(delayed|cancelled)|"
    r"security line at|boarding pass|layover|vacation (in|to|with)|back from (a |my |our )?\w+ trip",
    re.I,
)
STRICT_NEWS = re.compile(
    r"city council|council voted|public comment|mayor (says|said)|ballot measure|"
    r"primary turnout|record turnout|turnout in|school bond|hearings (on|opened|started)|"
    r"testified|cooling centers?|water main|grand jury|indict|verdict|city engineer|"
    r"(per|in) this morning's paper|state of emergency|evacuation order",
    re.I,
)
STRICT_FAMILY = re.compile(
    r"kindergarten (walk|drop|first)|first day of (school|kindergarten)|training wheels|"
    r"bedtime (negoti|routin|stall)|one more chapter|"
    r"my (daughter|son|kid|kids)\b|our (daughter|son|kids)\b|"
    r"my (mom|dad|mother|father|sister|brother)\b|called my (mom|dad)|"
    r"(she|he) is (four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen)\b|"
    r"at (mom|dad)'s|year-old.*(made (dinner|breakfast|lunch))",
    re.I,
)
STRICT_GAMES = re.compile(
    r"roguelike|deck-?builder|boss (fight|phase)|beat the (third |final )?boss|the run died|"
    r"floor one|hit points|last revive|one revive|respawn|speedrun|board game|tabletop|"
    r"session zero|the dm (asked|said|told)|dungeon master|campaign starts|puzzle game|"
    r"card game|loadout|nerf",
    re.I,
)
STRICT_ANIMALS = re.compile(
    r"songbirds?|outdoor cats|bird feeder|binoculars|birding|\btrail\b|hik(e|ing)|"
    r"\bdog\b|puppy|leash|wildlife|backyard",
    re.I,
)
STRICT_HEALTH = re.compile(
    r"protecting sleep|lights-out time|light goes off|phone charges in the kitchen|"
    r"massage gun|recovery (gadget|boot|gun)|cold plunge|protein|rest day|mobility|"
    r"hips|ankles|wellness|supplements|caffeine|no podcast",
    re.I,
)
STRICT_WRITING = re.compile(
    r"delete the first paragraph|throat-clearing|first paragraph is where|"
    r"drafting by hand|longhand|amwriting|shelf|translator",
    re.I,
)

STRICT = {
    "travel": STRICT_TRAVEL,
    "news": STRICT_NEWS,
    "family": STRICT_FAMILY,
    "games": STRICT_GAMES,
    "animals-nature": STRICT_ANIMALS,
    "health": STRICT_HEALTH,
    "writing-literature": STRICT_WRITING,
}


def lexicon_hits_v16_1(text: str) -> list[tuple[str, str]]:
    hits = lexicon_hits_v16(text)
    out: list[tuple[str, str]] = []
    for hint, hub in hits:
        gate = STRICT.get(hub)
        if gate is None or gate.search(text):
            out.append((hint, hub))
    return out


if __name__ == "__main__":
    # offline sanity: gold coverage must stay perfect on new hubs, 0 wrong first-hits
    from decision_bakeoff.data import load_labels, load_tsv

    labels = load_labels(REPO / "data" / "centa_topics_v16_otto_gold")
    rows = load_tsv(REPO / "data" / "centa_topics_v16_otto_gold" / "test.tsv")
    hits = 0
    wrong = []
    new_hub_cov: dict[str, list[int]] = {}
    for row in rows:
        gold = labels[int(row["label"])]
        h = lexicon_hits_v16_1(row["text"])
        if h and h[0][1] == gold:
            hits += 1
            if row["id"].startswith("cgold-synth"):
                new_hub_cov.setdefault(gold, [0, 0])
        if h and h[0][1] != gold:
            wrong.append((row["id"], gold, h[0]))
    for row in rows:
        gold = labels[int(row["label"])]
        if row["id"].startswith("cgold-synth"):
            new_hub_cov.setdefault(gold, [0, 0])
            new_hub_cov[gold][1] += 1
            h = lexicon_hits_v16_1(row["text"])
            if h and h[0][1] == gold:
                new_hub_cov[gold][0] += 1
    print("v16.1 gold first-hit correct", hits, "/", len(rows))
    print("new-hub coverage", {k: f"{v[0]}/{v[1]}" for k, v in sorted(new_hub_cov.items())})
    print("wrong first-hits", wrong)
