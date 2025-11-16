# kg_llm_generated.py

from __future__ import annotations
from typing import Dict, Any, List
import re
import requests

import streamlit as st
from kg_advanced import build_kg_advanced

# -------------------- LLM / Gemini config -------------------- #

GEMINI_MODEL = "gemini-2.5-flash"  # adjust if you use a different Gemini model
GEMINI_API_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)

# Max characters of combined corpus to send to the LLM
MAX_CORPUS_CHARS = 12000

# You can adjust these types if you have a domain-specific ontology
ENTITY_TYPES: List[str] = [
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "PRODUCT",
    "EVENT",
    "CONCEPT",
    "OTHER",
]

TUPLE_DELIMITER = "|||"
RECORD_DELIMITER = "\n"

def _build_prompt(texts: Dict[str, str]) -> str:
    """
    Build a single prompt string from all uploaded texts,
    along with the extraction instructions. Truncates the corpus
    to MAX_CORPUS_CHARS to avoid timeouts / overlong requests.
    """
    combined_parts: List[str] = []
    for name, content in texts.items():
        if not content:
            continue
        combined_parts.append(f"### SOURCE: {name}\n{content}")
    corpus_full = "\n\n".join(combined_parts)

    # Truncate corpus to keep request size reasonable
    corpus = corpus_full[:MAX_CORPUS_CHARS]

    entity_types_str = ", ".join(ENTITY_TYPES)

    instructions = f"""
You are an information extraction system that builds knowledge graphs from text.

Steps for Entity and Relationship Extraction:

1. Identify all entities. For each identified entity, extract the following information:
   - entity_name: Name of the entity, capitalized.
   - entity_type: One of the following specified types (e.g., [{entity_types_str}]).
   - entity_description: Comprehensive description of the entity’s attributes and activities.
   - Format: Format each entity as
     ("entity"{TUPLE_DELIMITER}<entity_name>{TUPLE_DELIMITER}<entity_type>{TUPLE_DELIMITER}<entity_description>).

2. Identify all pairs of related entities from the entities identified in Step 1 (source_entity, target_entity)
   that are clearly related to each other. For each pair, extract:
   - source_entity: name of the source entity, exactly as identified in step 1.
   - target_entity: name of the target entity, exactly as identified in step 1.
   - relationship_description: explanation as to why the source entity and target entity are related.
   - relationship_strength: a numeric score indicating the strength of the relationship (e.g., 0–10).
   - Format: Format each relationship as
     ("relationship"{TUPLE_DELIMITER}<source_entity>{TUPLE_DELIMITER}<target_entity>{TUPLE_DELIMITER}<relationship_description>{TUPLE_DELIMITER}<relationship_strength>).

3. Return Output:
   Return the output in English as a single list of all the entities and relationships identified in steps 1 and 2,
   using {repr(RECORD_DELIMITER)} as the list delimiter. Avoid using the tuple delimiter {repr(TUPLE_DELIMITER)} inside descriptions.

Only return this list. Do not add explanations or commentary.

TEXT TO ANALYZE:
{corpus}
""".strip()

    return instructions

# -------------------- Gemini call -------------------- #


def _call_gemini(prompt: str) -> str:
    """
    Call the Gemini endpoint with the provided prompt and return the text response.
    Uses GOOGLE_GEMINI_KEY from Streamlit secrets for authentication.
    """
    api_key = st.secrets.get("GOOGLE_GEMINI_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_GEMINI_KEY not found in Streamlit secrets.")

    params = {"key": api_key}
    headers = {"Content-Type": "application/json"}

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
        },
    }

    try:
        resp = requests.post(
            GEMINI_API_URL,
            headers=headers,
            params=params,
            json=payload,
            timeout=180,  # increase timeout to handle slower / larger requests
        )
    except requests.exceptions.ReadTimeout as e:
        raise RuntimeError(f"Gemini request timed out: {e}") from e

    resp.raise_for_status()
    data = resp.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        raise RuntimeError(f"Unexpected LLM response format: {e}")

# -------------------- Output parsing & kg_llm_generated stay unchanged -------------------- #
# (keep your existing _parse_llm_output and kg_llm_generated definitions)


def _parse_llm_output(raw: str) -> Dict[str, Any]:
    """
    Parse the LLM output into nodes, links, and evidence.
    Expected line format:
      ("entity"|||<name>|||<type>|||<description>)
      ("relationship"|||<src>|||<tgt>|||<description>|||<strength>)
    """
    nodes_by_name: Dict[str, Dict[str, Any]] = {}
    links: List[Dict[str, Any]] = []
    evidence_records: List[Dict[str, Any]] = []

    lines = [l.strip() for l in raw.split(RECORD_DELIMITER) if l.strip()]
    for line in lines:
        # Strip surrounding parentheses
        if line.startswith("(") and line.endswith(")"):
            line_inner = line[1:-1].strip()
        else:
            line_inner = line

        parts = [p.strip() for p in line_inner.split(TUPLE_DELIMITER)]
        if not parts:
            continue

        kind_raw = parts[0].strip()
        kind = kind_raw.strip('"').strip("'").lower()

        if kind == "entity":
            if len(parts) < 4:
                continue
            name = parts[1].strip().strip('"').strip("'")
            etype = parts[2].strip().strip('"').strip("'")
            desc = parts[3].strip().strip('"').strip("'")

            if not name:
                continue

            if name not in nodes_by_name:
                nodes_by_name[name] = {
                    "id": name,
                    "group": 3,  # distinct group id for LLM-based entities
                    "val": 1,
                    "type": etype,
                    "description": desc,
                }

        elif kind == "relationship":
            if len(parts) < 5:
                continue
            src = parts[1].strip().strip('"').strip("'")
            tgt = parts[2].strip().strip('"').strip("'")
            rel_desc = parts[3].strip().strip('"').strip("'")
            strength_raw = parts[4].strip().strip('"').strip("'")

            if not src or not tgt:
                continue

            try:
                strength_val = float(strength_raw)
            except ValueError:
                strength_val = 1.0

            links.append(
                {
                    "source": src,
                    "target": tgt,
                    "value": strength_val,
                }
            )
            evidence_records.append(
                {
                    "a": src,
                    "b": tgt,
                    "sentences": [rel_desc] if rel_desc else [],
                }
            )

    # Ensure all nodes referenced by relationships exist
    for link in links:
        for node_name in (link["source"], link["target"]):
            if node_name not in nodes_by_name:
                nodes_by_name[node_name] = {
                    "id": node_name,
                    "group": 3,
                    "val": 1,
                }

    nodes = list(nodes_by_name.values())
    return {"nodes": nodes, "links": links, "evidence": evidence_records}

# -------------------- Public entry point -------------------- #


def kg_llm_generated(texts: Dict[str, str]) -> Dict[str, Any]:
    """
    LLM-generated KG builder entry point.

    Takes:  texts: Dict[filename, content]
    Returns: {"nodes": [...], "links": [...], "evidence": [...] (optional)}
    """
    try:
        prompt = _build_prompt(texts)
        raw = _call_gemini(prompt)
        kg = _parse_llm_output(raw)

        # Safety fallback if parsing produced nothing
        if not kg.get("nodes") and not kg.get("links"):
            return build_kg_advanced(texts)
        return kg

    except Exception as e:
        st.error(f"LLM KG generation failed, falling back to advanced pipeline: {e}")
        return build_kg_advanced(texts)