# app.py
# NotebookLM-style Local Knowledge Graph App (Streamlit)
# ------------------------------------------------------
# Uses build_kg_basic (cleaned entities) + an external build_kg_advanced() from kg_advanced.py

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st

# Import your advanced KG pipeline from a separate module
from kg_advanced import build_kg_advanced

# ---------- Optional backends ----------
try:
    import pdfplumber  # pip: pdfplumber pypdfium2
except Exception:
    pdfplumber = None

try:
    import fitz  # PyMuPDF (conda: pymupdf)
except Exception:
    fitz = None

try:
    import docx  # python-docx
except Exception:
    docx = None

# Optional POS tagging (for removing ADJ/ADV precisely)
try:
    import spacy
    try:
        _nlp = spacy.load("en_core_web_sm")
    except Exception:
        _nlp = spacy.blank("en")
except Exception:
    spacy = None
    _nlp = None

# ---------- Page config ----------
st.set_page_config(page_title="NotebookLM-style KG Builder", page_icon="🧠", layout="wide")
SUPPORTED_EXTS = {".pdf", ".docx", ".txt", ".md"}

# ---------- Text extraction ----------
def extract_text_from_file(uploaded_file) -> str:
    name = uploaded_file.name
    suffix = Path(name).suffix.lower()
    data = uploaded_file.read()

    if suffix == ".pdf":
        from io import BytesIO
        if pdfplumber is not None:
            try:
                text_parts = []
                with pdfplumber.open(BytesIO(data)) as pdf:
                    for page in pdf.pages:
                        text_parts.append(page.extract_text() or "")
                return "\n".join(text_parts)
            except Exception:
                pass  # fall back to PyMuPDF
        if fitz is not None:
            try:
                doc = fitz.open(stream=BytesIO(data), filetype="pdf")
                text_parts = [page.get_text() or "" for page in doc]
                return "\n".join(text_parts)
            except Exception as e:
                return f"[Error reading PDF via PyMuPDF: {e}]"
        return "[No PDF backend available - install either with: pip install pdfplumber pypdfium2  OR  conda install -c conda-forge pymupdf]"

    if suffix == ".docx":
        if docx is None:
            return "[python-docx not installed]"
        from io import BytesIO
        d = docx.Document(BytesIO(data))
        return "\n".join(p.text for p in d.paragraphs)

    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return data.decode(errors="ignore")

# ---------- Entity cleaning utilities ----------
STOPWORDS = {
    "the","a","an","this","that","these","those","he","she","it","they","we","you","i",
    "his","her","their","its","our","your","my","me","him","them","us","mine","yours","ours","theirs",
    "of","in","on","at","by","for","with","about","against","between","into","through","during","before","after",
    "above","below","to","from","up","down","under","over","again","further","then","once","here","there","when",
    "where","why","how","all","any","both","each","few","more","most","other","some","such","nor","not","only",
    "own","same","so","than","too","very",
}

HEURISTIC_ADJ = {
    "new","old","former","current","other","various","several","many","much","great","good","bad",
    "small","large","big","major","minor","first","second","third","additional","different","same",
    "early","late","recent","past","present","future","global","local","international","national"
}

def looks_like_adverb(tok: str) -> bool:
    tok_l = tok.lower()
    return (len(tok_l) > 3 and tok_l.endswith("ly")) or tok_l in {
        "very","quite","rather","fairly","highly","truly","really","nearly","mostly","largely"
    }

def tokenize_capitalized_phrase(s: str) -> List[str]:
    raw = s.strip()
    tokens = [re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", t) for t in raw.split()]
    return [t for t in tokens if t]

def clean_entity_candidate(candidate: str) -> str:
    """
    - Trim leading/trailing stopwords (lowercased)
    - If spaCy with POS is available: keep only PROPN/NOUN/NUM tokens (drop ADJ/ADV/etc.)
    - Else: heuristically drop adjectives and adverbs tokens (and any pure stopwords)
    """
    toks = tokenize_capitalized_phrase(candidate)
    if not toks:
        return ""

    left = 0
    right = len(toks) - 1
    while left <= right and toks[left].lower() in STOPWORDS:
        left += 1
    while right >= left and toks[right].lower() in STOPWORDS:
        right -= 1
    toks = toks[left:right+1]
    if not toks:
        return ""

    if _nlp is not None and ("tagger" in _nlp.pipe_names or "morphologizer" in _nlp.pipe_names):
        doc = _nlp(" ".join(toks))
        kept = []
        for t in doc:
            if t.pos_ in {"PROPN","NOUN","NUM"}:
                if t.text.lower() in STOPWORDS:
                    continue
                kept.append(t.text)
        toks = kept
    else:
        kept = []
        for t in toks:
            tl = t.lower()
            if tl in STOPWORDS:
                continue
            if tl in HEURISTIC_ADJ or looks_like_adverb(tl):
                continue
            kept.append(t)
        toks = kept

    left = 0
    right = len(toks) - 1
    while left <= right and toks[left].lower() in STOPWORDS:
        left += 1
    while right >= left and toks[right].lower() in STOPWORDS:
        right -= 1
    toks = toks[left:right+1]

    return " ".join(toks) if toks else ""

# ---------- KG builders (basic) ----------
def build_kg_basic(texts: Dict[str, str]) -> Dict[str, Any]:
    """Baseline with stopword/adjective/adverb filtering + evidence capture."""
    corpus = []
    for _, t in texts.items():
        if not t:
            continue
        t = re.sub(r"\s+", " ", t)
        corpus.append(t)
    full_text = "\n".join(corpus)

    sentences = re.split(r"(?<=[.!?])\s+", full_text)
    entity_pattern = re.compile(r"(?:[A-Z][\w'\-]+(?:\s+[A-Z][\w'\-]+)+|[A-Z][\w'\-]+)")

    from collections import defaultdict
    node_counts = defaultdict(int)
    edge_counts = defaultdict(int)
    evidence = defaultdict(list)  # (a,b) -> [sentences]

    for sent in sentences:
        raw_ents = entity_pattern.findall(sent)
        raw_ents = [e.strip() for e in raw_ents if len(e.strip()) > 1]

        ents = []
        for e in raw_ents:
            ce = clean_entity_candidate(e)
            if not ce:
                continue
            if len(ce) < 2:
                continue
            if ce.lower() in STOPWORDS:
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

    nodes = [{"id": n, "group": 1, "val": max(1, c)} for n, c in node_counts.items()]
    links = [{"source": a, "target": b, "value": max(1, c)} for (a, b), c in edge_counts.items()]
    ev_list = [{"a": a, "b": b, "sentences": sents} for (a, b), sents in evidence.items()]

    return {"nodes": nodes, "links": links, "evidence": ev_list}

# ---------- 3D Renderer ----------
def render_3d_force_graph(graph_data: Dict[str, Any], height: int = 700):
    import streamlit.components.v1 as components
    data_json = json.dumps(graph_data)

    html_template = """
<!DOCTYPE html>
<html>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width, initial-scale=1'/>
  <style>
    html, body { height: 100%; margin: 0; background: #0b1020; }
    #container { position:absolute; left:0; top:0; right:360px; bottom:0; }
    #titlebar { position: absolute; top: 12px; left: 12px; color: #eaeefb; font-family: ui-sans-serif, system-ui; font-size: 14px; opacity: .9; z-index:10;}
    .pill { display:inline-block; background:#1b2340; padding:6px 10px; border-radius:999px; border:1px solid #2b365f }
    #inspector { position:absolute; right:0; top:0; width:360px; height:100%; background:#101736; border-left:1px solid #2b365f; color:#eaeefb; font-family: ui-sans-serif, system-ui; padding:14px; overflow:auto; }
  </style>
  <script src='https://unpkg.com/3d-force-graph@1.70.8/dist/3d-force-graph.min.js'></script>
  <script src='https://unpkg.com/three@0.160.0/build/three.min.js'></script>
</head>
<body>
  <div id='titlebar'><span class='pill'>Knowledge Graph (3D)</span></div>
  <div id='container'></div>
  <div id='inspector'>
    <div style='font-size:14px; opacity:.8; margin-bottom:8px;'>Inspector</div>
    <div id='inspector-content' style='font-size:14px; line-height:1.5;'>Select a node…</div>
  </div>
  <script>
    const data = __DATA__;

    const adj = new Map();
    const weight = new Map();
    (data.links || []).forEach(l => {
      const a = (typeof l.source === 'object') ? l.source.id : l.source;
      const b = (typeof l.target === 'object') ? l.target.id : l.target;
      if (!adj.has(a)) adj.set(a, new Set());
      if (!adj.has(b)) adj.set(b, new Set());
      adj.get(a).add(b);
      adj.get(b).add(a);
      const keyAB = a + '|||'+ b;
      const keyBA = b + '|||'+ a;
      weight.set(keyAB, (l.value || 1));
      weight.set(keyBA, (l.value || 1));
    });

    const evidence = new Map();
    (data.evidence || []).forEach(rec => {
      const a = rec.a, b = rec.b;
      const keyAB = a + '|||'+ b;
      const keyBA = b + '|||'+ a;
      evidence.set(keyAB, rec.sentences || []);
      evidence.set(keyBA, rec.sentences || []);
    });

    const elem = document.getElementById('container');
    const insp = document.getElementById('inspector-content');
    let selectedId = null;

    const Graph = ForceGraph3D()(elem)
      .graphData(data)
      .nodeId('id')
      .nodeVal('val')
      .nodeAutoColorBy('group')
      .nodeLabel(n => n.id)
      .backgroundColor('#0b1020')
      .linkColor(l => {
        const a = (typeof l.source === 'object') ? l.source.id : l.source;
        const b = (typeof l.target === 'object') ? l.target.id : l.target;
        if (!selectedId) return 'rgba(230,240,255,0.7)';
        return (a === selectedId || b === selectedId) ? 'rgba(255,255,255,0.95)' : 'rgba(230,240,255,0.15)';
      })
      .linkWidth(l => {
        const w = Math.max(1, Math.log2((l.value || 1) + 1));
        if (!selectedId) return w;
        const a = (typeof l.source === 'object') ? l.source.id : l.source;
        const b = (typeof l.target === 'object') ? l.target.id : l.target;
        return (a === selectedId || b === selectedId) ? w + 1.2 : w;
      })
      .linkOpacity(0.7)
      .linkDirectionalParticles(2)
      .linkDirectionalParticleSpeed(0.006)
      .onNodeClick(node => {
        selectedId = node.id;
        updateInspector(node);
        Graph.linkColor(Graph.linkColor());
        Graph.linkWidth(Graph.linkWidth());
      })
      .onNodeHover(n => elem.style.cursor = n ? 'pointer' : null)
      .nodeThreeObject(node => {
        const group = new THREE.Group();
        const size = Math.max(3, Math.cbrt(node.val || 1) * 3);
        const geometry = new THREE.SphereGeometry(size, 16, 16);
        const material = new THREE.MeshPhongMaterial({ color: 0x88ccff, shininess: 30 });
        const sphere = new THREE.Mesh(geometry, material);
        group.add(sphere);

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const fontSize = 38;
        ctx.font = fontSize + 'px sans-serif';
        const text = String(node.id || '');
        const padding = 20;
        const w = ctx.measureText(text).width + padding * 2;
        const h = fontSize + padding * 2;
        canvas.width = w;
        canvas.height = h;
        const ctx2 = canvas.getContext('2d');
        ctx2.font = fontSize + 'px sans-serif';
        ctx2.fillStyle = 'rgba(15,20,40,0.8)';
        ctx2.fillRect(0, 0, w, h);
        ctx2.fillStyle = 'white';
        ctx2.textBaseline = 'middle';
        ctx2.fillText(text, padding, h / 2);

        const texture = new THREE.CanvasTexture(canvas);
        texture.minFilter = THREE.LinearFilter;
        const spriteMat = new THREE.SpriteMaterial({ map: texture, depthTest: false });
        const sprite = new THREE.Sprite(spriteMat);
        sprite.scale.set(w / 8, h / 8, 1);
        sprite.position.set(0, size + (h / 16) + 2, 0);
        group.add(sprite);

        return group;
      });

    function updateInspector(node) {
      const id = node.id;
      const neighbors = Array.from(adj.get(id) || []).sort((a,b) =>
        (weight.get(id+'|||'+b)||0) - (weight.get(id+'|||'+a)||0)
      ).reverse();

      let html = '<div style="font-size:18px;font-weight:600;margin-bottom:6px;">' + id + '</div>';
      html += '<div style="opacity:.8;margin-bottom:10px;">Degree: ' + neighbors.length + ' • Mentions: ' + (node.val||1) + '</div>';

      if (neighbors.length === 0) {
        html += '<div style="opacity:.8">No connected neighbors.</div>';
        insp.innerHTML = html;
        return;
      }

      html += '<div style="margin:8px 0 6px; font-weight:600;">Neighbors</div>';
      neighbors.slice(0, 50).forEach(nbr => {
        const w = weight.get(id + '|||'+ nbr) || 1;
        const ev = evidence.get(id + '|||'+ nbr) || [];
        html += '<div style="margin:6px 0; padding:6px; background:#0f1430; border:1px solid #28325a; border-radius:8px;">' +
                '<div style="display:flex;justify-content:space-between;align-items:center;">' +
                '<div><span style="font-weight:600;">' + nbr + '</span></div>' +
                '<div style="opacity:.8;">w=' + w + '</div>' +
                '</div>';
        if (ev.length) {
          html += '<ul style="margin:6px 0 0 16px;padding:0;">' +
                  ev.slice(0,2).map(s => '<li style="margin:4px 0;">' + s + '</li>').join('') +
                  '</ul>';
        }
        html += '</div>';
      });
      insp.innerHTML = html;
    }
  </script>
</body>
</html>
    """

    html = html_template.replace("__DATA__", data_json)
    import streamlit.components.v1 as components
    components.html(html, height=height)

# ---------- Sidebar / Controls ----------
with st.sidebar:
    st.markdown("## 📚 Sources")
    uploaded = st.file_uploader(
        "Upload documents (PDF, DOCX, TXT, MD)",
        type=[e.strip(".") for e in SUPPORTED_EXTS],
        accept_multiple_files=True,
        help="Add multiple files; they'll be combined for KG construction.",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Graph Options")
    min_freq = st.slider("Minimum entity frequency", 1, 5, 1, 1)
    min_link = st.slider("Minimum link weight", 1, 5, 1, 1)
    graph_height = st.slider("Graph canvas height (px)", 400, 1200, 720, 20)

    st.markdown("---")
    st.markdown("### 🚀 Actions")
    gen_adv = st.button("Generate Knowledge Graph", type="primary")
    gen_basic = st.button("KG-Graph basic")

    st.markdown("---")
    st.markdown("### 🧪 Diagnostics")
    st.caption(
        f"pdfplumber: {'✅' if pdfplumber is not None else '❌'} • "
        f"PyMuPDF: {'✅' if fitz is not None else '❌'} • "
        f"DOCX: {'✅' if docx is not None else '❌'} • "
        f"spaCy: {'✅' if _nlp is not None else '❌'}"
    )

# ---------- Main header ----------
st.markdown("""
# 🧠 NotebookLM-style KG Builder (web version)
Upload a set of sources on the left, then build a 3D knowledge graph from them. Use **Generate Knowledge Graph** for your advanced pipeline (external module) or try **KG-Graph basic** for a simple baseline.
""")

# ---------- Read files and preview ----------
texts: Dict[str, str] = {}
if uploaded:
    st.markdown("## 📄 Selected Sources")
    cols = st.columns(min(3, len(uploaded)))
    for i, uf in enumerate(uploaded):
        with cols[i % len(cols)]:
            st.caption(f"**{uf.name}**")
            try:
                uf.seek(0)
                content = extract_text_from_file(uf)
                texts[uf.name] = content
                preview = (content or "").strip()[:500]
                st.code(preview + ("…" if len(content) > 500 else ""), language="markdown")
            except Exception as e:
                st.error(f"Failed to read {uf.name}: {e}")
else:
    st.info("Upload a few files to get started.")

# ---------- Session state ----------
if "graph_data" not in st.session_state:
    st.session_state["graph_data"] = None

# ---------- Actions ----------
if (gen_adv or gen_basic) and not texts:
    st.warning("Please upload at least one document first.")

if gen_adv and texts:
    with st.spinner("Building advanced knowledge graph (external pipeline)…"):
        g = build_kg_advanced(texts)
        st.session_state["graph_data"] = g

if gen_basic and texts:
    with st.spinner("Building basic co-occurrence graph…"):
        g = build_kg_basic(texts)
        st.session_state["graph_data"] = g

# ---------- Threshold + render ----------
if st.session_state["graph_data"]:
    g = st.session_state["graph_data"]
    nodes = g.get("nodes", [])
    links = g.get("links", [])

    keep_nodes = {n["id"] for n in nodes if n.get("val", 1) >= min_freq}
    filt_links = [
        L for L in links
        if L.get("value", 1) >= min_link and L["source"] in keep_nodes and L["target"] in keep_nodes
    ]

    referenced = set()
    for L in filt_links:
        referenced.add(L["source"])
        referenced.add(L["target"])
    filt_nodes = [n for n in nodes if n["id"] in referenced]

    filtered_graph = {"nodes": filt_nodes, "links": filt_links}

    if "evidence" in g:
        filtered_graph["evidence"] = [
            ev for ev in g["evidence"] if ev["a"] in referenced and ev["b"] in referenced
        ]

    st.markdown("## 🕸️ Knowledge Graph (3D)")
    render_3d_force_graph(filtered_graph, height=graph_height)

    colA, colB = st.columns(2)
    with colA:
        st.download_button(
            label="⬇️ Export KG as JSON",
            file_name="knowledge_graph.json",
            mime="application/json",
            data=json.dumps(filtered_graph, indent=2),
        )
    with colB:
        st.caption("Tip: Adjust thresholds in the sidebar to declutter the graph.")
