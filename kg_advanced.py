# kg_advanced.py
# -----------------------------------------
# Advanced KG construction pipeline stub.
# The main app imports build_kg_advanced() from here.
#
# Replace the internals of build_kg_advanced() with your
# real algorithm (LLM, NER/RE, tensor methods, etc.).
#
# Contract:
#   build_kg_advanced(texts: Dict[str, str]) -> Dict[str, Any]
#   returns at least:
#       {"nodes": [...], "links": [...]}
#   and optionally:
#       "evidence": [{"a": str, "b": str, "sentences": [str, ...]}, ...]
#
# The UI / visualizer expects:
#   nodes: {"id": <entity_id>, "group": int, "val": numeric weight}
#   links: {"source": <id>, "target": <id>, "value": numeric weight}
#
# Placeholder for future function

from __future__ import annotations
from typing import Dict, Any
import re
from collections import defaultdict

# -------------------- Heuristic lexical resources -------------------- #

STOPWORDS = {
    "the", "a", "an", "this", "that", "these", "those",
    "of", "in", "on", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "under", "over",
    "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very",
}

PRONOUNS = {
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them",
    "my", "your", "his", "her", "its", "our", "their",
    "mine", "yours", "ours", "theirs", "hers",
}

# Common adjectives that often appear in capitalized phrases
HEURISTIC_ADJ = {
    "new", "old", "former", "current", "other", "various",
    "several", "many", "much", "great", "good", "bad",
    "small", "large", "big", "major", "minor",
    "first", "second", "third", "additional", "different", "same",
    "early", "late", "recent", "past", "present", "future",
    "global", "local", "international", "national",
}

# Common verbs we want to avoid as “entities” (even if capitalized)
HEURISTIC_VERBS = {
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "make", "makes", "made", "go", "goes", "went",
    "say", "says", "said", "get", "gets", "got",
    "know", "knows", "knew", "think", "thinks", "thought",
}

def _looks_like_adverb(tok: str) -> bool:
    tok_l = tok.lower()
    return (len(tok_l) > 3 and tok_l.endswith("ly")) or tok_l in {
        "very", "quite", "rather", "fairly", "highly",
        "truly", "really", "nearly", "mostly", "largely",
    }

def _tokenize_capitalized_phrase(s: str) -> List[str]:
    """Split a phrase and strip punctuation at token boundaries."""
    raw = s.strip()
    tokens = [re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", t) for t in raw.split()]
    return [t for t in tokens if t]

def _clean_entity_candidate(candidate: str) -> str:
    """
    Heuristically clean a capitalized phrase so it becomes a better entity:
    - Split into tokens
    - Remove stopwords, pronouns, adjectives, adverbs, and common verbs
    - Trim any remaining leading/trailing stopwords/pronouns
    """
    toks = _tokenize_capitalized_phrase(candidate)
    if not toks:
        return ""

    kept: List[str] = []
    for t in toks:
        tl = t.lower()

        # Drop obvious function words/pronouns
        if tl in STOPWORDS or tl in PRONOUNS:
            continue

        # Drop adjectives/adverbs
        if tl in HEURISTIC_ADJ or _looks_like_adverb(tl):
            continue

        # Drop very common verbs
        if tl in HEURISTIC_VERBS:
            continue

        kept.append(t)

    if not kept:
        return ""

    # Trim any remaining leading/trailing stopwords/pronouns just in case
    left = 0
    right = len(kept) - 1
    while left <= right and kept[left].lower() in STOPWORDS.union(PRONOUNS):
        left += 1
    while right >= left and kept[right].lower() in STOPWORDS.union(PRONOUNS):
        right -= 1
    kept = kept[left:right + 1]

    return " ".join(kept) if kept else ""

# -------------------- Advanced KG builder (no spaCy) -------------------- #

def build_kg_advanced(texts: Dict[str, str]) -> Dict[str, Any]:
    """
    Advanced KG builder WITHOUT spaCy:
    - Extracts capitalized phrases as entity candidates.
    - Heuristically removes common words, pronouns, adjectives, adverbs, verbs.
    - Builds:
        * nodes: entities with frequency counts
        * links: co-occurrence in the same sentence
        * evidence: example sentences per edge
    """
    # 1) Combine and normalize corpus
    corpus: List[str] = []
    for _, t in texts.items():
        if not t:
            continue
        t = re.sub(r"\s+", " ", t)
        corpus.append(t)
    full_text = "\n".join(corpus)

    if not full_text.strip():
        return {"nodes": [], "links": [], "evidence": []}

    # 2) Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", full_text)

    # Regex: capitalized phrases (single or multi-word)
    entity_pattern = re.compile(
        r"(?:[A-Z][\w'\-]+(?:\s+[A-Z][\w'\-]+)+|[A-Z][\w'\-]+)"
    )

    node_counts = defaultdict(int)
    edge_counts = defaultdict(int)
    evidence = defaultdict(list)  # (a, b) -> [sentences]

    # 3) Extract & clean entities, then count co-occurrences
    for sent in sentences:
        if not sent.strip():
            continue

        raw_ents = entity_pattern.findall(sent)
        raw_ents = [e.strip() for e in raw_ents if len(e.strip()) > 1]

        ents: List[str] = []
        for e in raw_ents:
            ce = _clean_entity_candidate(e)
            if not ce:
                continue
            if len(ce) < 2:
                continue
            # extra guard: do not treat leftover stopwords/pronouns as entities
            if ce.lower() in STOPWORDS or ce.lower() in PRONOUNS:
                continue
            ents.append(ce)

        # Deduplicate within sentence
        uniq = sorted(set(ents))

        # Count nodes: each unique entity per sentence increments its count
        for e in uniq:
            node_counts[e] += 1

        # Count edges: co-occurrence within the sentence
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                a, b = uniq[i], uniq[j]
                edge_counts[(a, b)] += 1
                # Keep up to a few example sentences as evidence
                if len(evidence[(a, b)]) < 3:
                    snippet = sent.strip()
                    if len(snippet) > 240:
                        snippet = snippet[:240] + "…"
                    evidence[(a, b)].append(snippet)

    # 4) Build graph structure
    nodes = [
        {"id": n, "group": 2, "val": max(1, c)}
        for n, c in node_counts.items()
    ]
    links = [
        {"source": a, "target": b, "value": max(1, c)}
        for (a, b), c in edge_counts.items()
    ]
    ev_list = [
        {"a": a, "b": b, "sentences": sents}
        for (a, b), sents in evidence.items()
    ]

    return {"nodes": nodes, "links": links, "evidence": ev_list}
