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
from typing import Dict, Any, List, Tuple
import re
from collections import defaultdict
# Optional: use spaCy if available for NER + POS tagging
try:
    import spacy
    try:
        _nlp = spacy.load("en_core_web_sm")
    except Exception:
        # Fallback to blank English pipeline if model not installed
        _nlp = spacy.blank("en")
except Exception:
    spacy = None
    _nlp = None

# -------------------- Text / entity cleaning helpers -------------------- #

STOPWORDS = {
    "the", "a", "an", "this", "that", "these", "those",
    "he", "she", "it", "they", "we", "you", "i",
    "his", "her", "their", "its", "our", "your", "my",
    "me", "him", "them", "us", "mine", "yours", "ours", "theirs",
    "of", "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "under", "over",
    "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very",
}

HEURISTIC_ADJ = {
    "new", "old", "former", "current", "other", "various", "several",
    "many", "much", "great", "good", "bad",
    "small", "large", "big", "major", "minor",
    "first", "second", "third", "additional", "different", "same",
    "early", "late", "recent", "past", "present", "future",
    "global", "local", "international", "national",
}

PRONOUNS = {
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them",
    "my", "your", "his", "her", "its", "our", "their",
    "mine", "yours", "ours", "theirs", "hers",
}

def _looks_like_adverb(tok: str) -> bool:
    tok_l = tok.lower()
    return (len(tok_l) > 3 and tok_l.endswith("ly")) or tok_l in {
        "very", "quite", "rather", "fairly", "highly", "truly", "really",
        "nearly", "mostly", "largely",
    }

def _tokenize_capitalized_phrase(s: str) -> List[str]:
    raw = s.strip()
    tokens = [re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", t) for t in raw.split()]
    return [t for t in tokens if t]

def _clean_entity_heuristic(candidate: str) -> str:
    """
    Heuristic cleaner used when spaCy NER is not available:
    - split capitalized phrase into tokens
    - remove stopwords, pronouns, adjectives/adverbs
    """
    toks = _tokenize_capitalized_phrase(candidate)
    if not toks:
        return ""

    kept = []
    for t in toks:
        tl = t.lower()
        if tl in STOPWORDS or tl in PRONOUNS:
            continue
        if tl in HEURISTIC_ADJ or _looks_like_adverb(tl):
            continue
        kept.append(t)

    # Remove leading/trailing stopwords/pronouns again just in case
    left = 0
    right = len(kept) - 1
    while left <= right and kept[left].lower() in STOPWORDS.union(PRONOUNS):
        left += 1
    while right >= left and kept[right].lower() in STOPWORDS.union(PRONOUNS):
        right -= 1
    kept = kept[left:right + 1]

    return " ".join(kept) if kept else ""

def _extract_entities_spacy(text: str) -> List[Tuple[str, str]]:
    """
    Return list of (entity_text, sentence_text) from spaCy NER,
    filtering out pronouns/stopwords/verbs via POS where possible.
    """
    if _nlp is None:
        return []

    doc = _nlp(text)
    results: List[Tuple[str, str]] = []

    # Map spaCy span to its sentence text
    for sent in doc.sents if doc.has_annotation("SENT_START") else [doc]:
        for ent in sent.ents:
            # Keep only named entities that tend to be meaningful in a KG
            if ent.label_ not in {"PERSON", "ORG", "GPE", "LOC", "EVENT", "PRODUCT", "WORK_OF_ART"}:
                continue

            # Filter tokens inside entity: remove stopwords, pronouns, verbs, adverbs, etc.
            kept_tokens = []
            for t in ent:
                tl = t.text.lower()
                if tl in STOPWORDS or tl in PRONOUNS:
                    continue
                if t.pos_ in {"VERB", "AUX", "ADV", "ADP", "DET", "PRON"}:
                    continue
                if _looks_like_adverb(tl):
                    continue
                kept_tokens.append(t.text)

            cleaned = " ".join(kept_tokens).strip()
            if len(cleaned) < 2:
                continue
            results.append((cleaned, sent.text.strip()))

    return results

# -------------------- Main builder -------------------- #

def build_kg_advanced(texts: Dict[str, str]) -> Dict[str, Any]:
    """
    Advanced KG builder:
    - If spaCy is available: use NER + POS to keep nouns/proper nouns and drop
      stopwords, pronouns, verbs, adverbs, etc.
    - Otherwise: fall back to capitalized-phrase regex + heuristic cleaning.
    - Builds nodes (entities) and links (co-occurrence within a sentence).
    - Also returns up to a few example sentences as 'evidence' per edge.
    """
    # --------- 1. Combine and normalize corpus --------- #
    corpus: List[str] = []
    for _, t in texts.items():
        if not t:
            continue
        # collapse whitespace
        t = re.sub(r"\s+", " ", t)
        corpus.append(t)
    full_text = "\n".join(corpus)

    # Quick exit if empty
    if not full_text.strip():
        return {"nodes": [], "links": [], "evidence": []}

    # Split into rough sentences (used in fallback mode and also in evidence)
    sentences = re.split(r"(?<=[.!?])\s+", full_text)

    # --------- 2. Entity extraction --------- #
    node_counts = defaultdict(int)
    edge_counts = defaultdict(int)
    evidence = defaultdict(list)  # (a, b) -> [sentences]

    # Case A: spaCy NER available: work directly on full_text
    if _nlp is not None:
        ents_with_sent = _extract_entities_spacy(full_text)

        # Group entities by sentence text
        sent_to_ents: defaultdict[str, List[str]] = defaultdict(list)
        for ent_text, sent_text in ents_with_sent:
            sent_to_ents[sent_text].append(ent_text)

        for sent_text, ents in sent_to_ents.items():
            # Normalize and deduplicate entities for this sentence
            norm_ents = []
            for e in ents:
                ce = e.strip()
                if not ce:
                    continue
                if len(ce) < 2:
                    continue
                if ce.lower() in STOPWORDS or ce.lower() in PRONOUNS:
                    continue
                norm_ents.append(ce)

            uniq = sorted(set(norm_ents))
            # Count nodes
            for e in uniq:
                node_counts[e] += 1
            # Count edges + evidence
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    a, b = uniq[i], uniq[j]
                    edge_counts[(a, b)] += 1
                    if len(evidence[(a, b)]) < 3:
                        snippet = sent_text
                        if len(snippet) > 240:
                            snippet = snippet[:240] + "…"
                        evidence[(a, b)].append(snippet)

    # Case B: no spaCy NER – fall back to regex + heuristics
    else:
        entity_pattern = re.compile(
            r"(?:[A-Z][\w'\-]+(?:\s+[A-Z][\w'\-]+)+|[A-Z][\w'\-]+)"
        )

        for sent in sentences:
            raw_ents = entity_pattern.findall(sent)
            raw_ents = [e.strip() for e in raw_ents if len(e.strip()) > 1]

            ents: List[str] = []
            for e in raw_ents:
                ce = _clean_entity_heuristic(e)
                if not ce:
                    continue
                if len(ce) < 2:
                    continue
                if ce.lower() in STOPWORDS or ce.lower() in PRONOUNS:
                    continue
                ents.append(ce)

            uniq = sorted(set(ents))
            for e in uniq:
                node_counts[e] += 1

            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    a, b = uniq[i], uniq[j]
                    edge_counts[(a, b)] += 1
                    if len(evidence[(a, b)]) < 3:
                        snippet = sent.strip()
                        if len(snippet) > 240:
                            snippet = snippet[:240] + "…"
                        evidence[(a, b)].append(snippet)

    # --------- 3. Build node/link lists --------- #
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
