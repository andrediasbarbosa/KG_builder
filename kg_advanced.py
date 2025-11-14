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

def build_kg_advanced(texts: Dict[str, str]) -> Dict[str, Any]:
    """
    Placeholder implementation:
    - Simple capitalized-entity co-occurrence (no stopword/adj/adv cleaning).
    - You should replace this with your advanced pipeline.

    Args:
        texts: mapping filename -> raw text

    Returns:
        dict with "nodes" and "links" (and optionally "evidence").
    """
    corpus = []
    for _, t in texts.items():
        if not t:
            continue
        t = re.sub(r"\s+", " ", t)
        corpus.append(t)
    full_text = "\n".join(corpus)

    sentences = re.split(r"(?<=[.!?])\s+", full_text)

    entity_pattern = re.compile(r"(?:[A-Z][\w'\-]+(?:\s+[A-Z][\w'\-]+)+|[A-Z][\w'\-]+)")

    node_counts = defaultdict(int)
    edge_counts = defaultdict(int)

    for sent in sentences:
        ents = entity_pattern.findall(sent)
        ents = [e.strip() for e in ents if len(e.strip()) > 1]
        uniq = sorted(set(ents))
        for e in uniq:
            node_counts[e] += 1
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                a, b = uniq[i], uniq[j]
                edge_counts[(a, b)] += 1

    nodes = [{"id": n, "group": 2, "val": max(1, c)} for n, c in node_counts.items()]
    links = [{"source": a, "target": b, "value": max(1, c)} for (a, b), c in edge_counts.items()]

    # You can also attach evidence here similar to the basic builder:
    # evidence = [...]
    # return {"nodes": nodes, "links": links, "evidence": evidence}

    return {"nodes": nodes, "links": links}
