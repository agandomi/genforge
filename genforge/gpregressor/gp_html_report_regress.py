# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi

import os
import html
from typing import List, Tuple, Any

################################################################################
# Public API
################################################################################

def write_ensemble_html(
    gp,
    ensemble_row: int,
    out_path: str,
    title: str = "GenForge Regression Ensemble Report",
) -> str:
    """
    Create a self-contained HTML file visualizing a chosen regression ensemble.

    Expected gp fields (already evaluated):
      - gp.population[pop][ind] -> List[str] gene strings
      - gp.config['runcontrol']['num_pop'], gp.config['runcontrol']['minimisation']
      - gp.individuals['ensemble_idx'][ensemble_row, pop] -> int member index
      - (opt) gp.individuals['ensemble_weight'][ensemble_row] -> (P,) or (P,T)
      - (opt) gp.individuals['weight_genes'][pop][ind] -> linear head:
              shape (T, G+1) (last col bias) OR (G+1, T) (last row bias)
              OR (G+1,) for single-target
      - (opt) gp.individuals['fitness']['isolated'][train|validation|test] -> (N,P)
      - (opt) gp.individuals['fitness']['ensemble'][train|validation|test] -> (E,)
    """
    # --- ensemble membership ---------------------------------------------------
    num_pop = int(gp.config['runcontrol']['num_pop'])
    ens_idx = _safe_get(gp.individuals, ['ensemble_idx'])
    if ens_idx is None:
        raise RuntimeError("gp.individuals['ensemble_idx'] not found. Run ensemble evaluation first.")
    if ensemble_row < 0 or ensemble_row >= ens_idx.shape[0]:
        raise IndexError(f"ensemble_row {ensemble_row} out of range [0, {ens_idx.shape[0]-1}]")

    selected_by_pop = [(p, int(ens_idx[ensemble_row, p])) for p in range(num_pop)]

    # Optional data
    ensemble_weight = _safe_get(gp.individuals, ['ensemble_weight'])
    weight_genes    = _safe_get(gp.individuals, ['weight_genes'])

    # --- sections per (pop, ind) ----------------------------------------------
    member_sections = []
    num_targets = _get_num_targets(gp)  # None or int

    for p, ind in selected_by_pop:
        # gene strings for this individual
        genes = list(gp.population[p][ind])
        ind_title = f"Population {p} · Individual {ind}"

        # linear head weights for this individual
        WG = None
        try:
            WG = weight_genes[p][ind]
        except Exception:
            WG = None

        weights_table_html, yhat_equations_html = _format_linreg_weights_table(WG, num_targets)

        # Render trees
        trees_html = []
        for g_idx, gene_str in enumerate(genes, 1):
            node = parse_gene_string(gene_str)
            svg = render_tree_svg(node, node_radius=16, hgap=20, vgap=60)
            trees_html.append(f"""
            <div class="gene-card">
              <div class="gene-title">Gene {g_idx}</div>
              <div class="gene-code"><code>{html.escape(gene_str)}</code></div>
              <div class="gene-svg">{svg}</div>
            </div>
            """)

        # per-member fitness + rank
        member_fit_html = _format_member_fitness_block(gp, p, ind)

        member_sections.append(f"""
        <section class="member">
          <h3>{html.escape(ind_title)}</h3>
          {member_fit_html}
          <details open>
            <summary>Gene trees</summary>
            <div class="gene-grid">
              {''.join(trees_html)}
            </div>
          </details>
          {yhat_equations_html}
          {weights_table_html}
        </section>
        """)

    # --- ensemble weights table (if any) --------------------------------------
    ensemble_table_html = ""
    if ensemble_weight is not None:
        ensemble_table_html = _format_ensemble_weight_table_regress(
            ensemble_weight, ensemble_row, num_pop, num_targets
        )

    # --- ensemble fitness + ranking -------------------------------------------
    ensemble_fit_html = _format_ensemble_fitness_block(gp, ensemble_row)

    # --- compose & write ------------------------------------------------------
    html_text = _wrap_html(
        title=title,
        header=_header_block(title),
        ensemble_table=ensemble_table_html,
        ensemble_fitness=ensemble_fit_html,
        members="".join(member_sections),
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_text)
    return os.path.abspath(out_path)

################################################################################
# Parsing gene strings -> simple AST
################################################################################

class Node:
    __slots__ = ("label", "children")
    def __init__(self, label: str, children: List["Node"]=None):
        self.label = label
        self.children = children or []

def parse_gene_string(s: str) -> Node:
    """
    Parse:
      'plus(x1,times(x2,[3.0]))', 'times(x1,x2)', 'x1', '[1.23]', 'rand()'
    into a Node tree. Brackets are kept in the AST so we can strip visually.
    """
    s = s.strip()

    def parse_expr(i: int) -> Tuple[Node, int]:
        # constant like [1.23]
        if i < len(s) and s[i] == '[':
            j = i + 1
            while j < len(s) and s[j] != ']':
                j += 1
            if j >= len(s):
                raise ValueError("Unclosed constant '['")
            val = s[i:j+1]
            return Node(val, []), j + 1

        # variable like x12
        if i < len(s) and s[i] == 'x':
            j = i + 1
            while j < len(s) and s[j].isdigit():
                j += 1
            return Node(s[i:j], []), j

        # function token
        j = i
        while j < len(s) and (s[j].isalnum() or s[j] == '_' or s[j] == '$'):
            j += 1
        name = s[i:j]
        if j >= len(s) or s[j] != '(':
            return Node(name, []), j

        # args
        j += 1
        args = []
        if j < len(s) and s[j] == ')':
            j += 1
            return Node(name, []), j

        start = j
        depth = 0
        for k in range(j, len(s)):
            c = s[k]
            if c == '(' or c == '[':
                depth += 1
            elif c == ')' or c == ']':
                depth -= 1
            if (c == ',' and depth == 0) or (c == ')' and depth == -1):
                sub = s[start:k] if c == ',' else s[start:k]
                node, _ = parse_expr_inner(sub)
                args.append(node)
                start = k + 1
            if c == ')' and depth == -1:
                j = k + 1
                break

        return Node(name, args), j

    def parse_expr_inner(chunk: str) -> Tuple[Node, int]:
        return parse_gene_string(chunk), len(chunk)

    node, pos = parse_expr(0)
    if pos != len(s):
        rest = s[pos:].strip().strip(',')
        if rest:
            raise ValueError(f"Trailing input while parsing: {rest!r}")
    return node

################################################################################
# Tree layout and SVG rendering
################################################################################

def _measure(node: Node, node_radius=16, hgap=20) -> int:
    if not node.children:
        return 2 * node_radius
    widths = [_measure(ch, node_radius, hgap) for ch in node.children]
    return max(2 * node_radius, sum(widths) + hgap * (len(widths) - 1))

def _layout(node: Node, x: float, y: float, node_radius=16, hgap=20, vgap=60):
    """
    Returns (nodes, edges, width) with:
      nodes = [(label, x, y)]
      edges = [(x1,y1, x2,y2)]
    """
    width = _measure(node, node_radius, hgap)
    nodes = []
    edges = []
    if not node.children:
        nodes.append((node.label, x, y))
        return nodes, edges, width

    child_widths = [_measure(ch, node_radius, hgap) for ch in node.children]
    total = sum(child_widths) + hgap * (len(child_widths) - 1)
    cx = x - total / 2.0

    nodes.append((node.label, x, y))
    for ch, w in zip(node.children, child_widths):
        child_x = cx + w / 2.0
        child_y = y + vgap
        cnodes, cedges, _ = _layout(ch, child_x, child_y, node_radius, hgap, vgap)
        nodes.extend(cnodes)
        edges.extend(cedges)
        edges.append((x, y, child_x, child_y))
        cx += w + hgap
    return nodes, edges, width

def render_tree_svg(node: Node, node_radius=16, hgap=20, vgap=60) -> str:
    """
    Inline SVG tree with:
      - page bg:      set in CSS (#f3f4f6)
      - tree bg:      #e5e7eb
      - edges:        black
      - nodes:        light blue, black stroke
    """
    nodes, edges, width = _layout(node, x=0, y=0, node_radius=node_radius, hgap=hgap, vgap=vgap)

    xs = [x for _, x, _ in nodes] + [x1 for x1,_,x2,_ in edges] + [x2 for _,_,x2,_ in edges]
    ys = [y for _, _, y in nodes] + [y1 for _,y1,_,_ in edges] + [y2 for _,_,_,y2 in edges]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    pad = 24
    shift_x = -minx + pad
    shift_y = -miny + pad
    W = int(maxx - minx + 2 * pad)
    H = int(maxy - miny + 2 * pad)

    svg_bg     = "#e5e7eb"
    edge_color = "#000000"
    node_fill  = "#cfe8ff"
    node_stroke= "#000000"
    text_fill  = "#000000"

    out = []
    out.append(f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Gene tree">')
    out.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="{svg_bg}"/>')
    for (x1, y1, x2, y2) in edges:
        out.append(f'<line x1="{x1+shift_x:.1f}" y1="{y1+shift_y:.1f}" x2="{x2+shift_x:.1f}" y2="{y2+shift_y:.1f}" stroke="{edge_color}" stroke-width="1.5"/>')
    for (label, x, y) in nodes:
        cx, cy = x + shift_x, y + shift_y
        lab = str(label)
        if len(lab) >= 2 and lab[0] == "[" and lab[-1] == "]":
            lab = lab[1:-1]  # strip brackets visually
        safe = html.escape(lab)
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{node_radius}" fill="{node_fill}" stroke="{node_stroke}" stroke-width="1.2"/>')
        out.append(f'<text x="{cx:.1f}" y="{cy+4:.1f}" font-size="11" text-anchor="middle" fill="{text_fill}" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial">{safe}</text>')
    out.append("</svg>")
    return "\n".join(out)

################################################################################
# Fitness / Ranks helpers
################################################################################

def _format_member_fitness_block(gp, pop_idx: int, ind_idx: int) -> str:
    """Small table: train/val/test fitness and rank of an isolated individual within its population."""
    try:
        import numpy as np
    except Exception:
        return ""

    minim = bool(gp.config['runcontrol'].get('minimisation', False))
    datasets = ['train', 'validation', 'test']
    rows_html = []
    any_present = False

    for ds in datasets:
        arr = _safe_get(gp.individuals, ['fitness','isolated', ds])
        if arr is None:
            continue
        try:
            M = np.asarray(arr, dtype=float)  # (num_ind, num_pop)
            val = float(M[ind_idx, pop_idx])
            order = np.argsort(M[:, pop_idx]) if minim else np.argsort(-M[:, pop_idx])
            rank = int(np.where(order == ind_idx)[0][0]) + 1
            total = M.shape[0]
            rows_html.append(f"<tr><td>{ds}</td><td>{val:.6g}</td><td>{rank} / {total}</td></tr>")
            any_present = True
        except Exception:
            continue

    if not any_present:
        return ""

    return f"""
    <section class="metrics">
      <h4>Isolated fitness &amp; rank</h4>
      <div class="table-wrap tight">
        <table class="latex">
          <tr><th>Split</th><th>Fitness</th><th>Rank in pop {pop_idx}</th></tr>
          {''.join(rows_html)}
        </table>
      </div>
    </section>
    """

def _format_ensemble_fitness_block(gp, ensemble_row: int) -> str:
    """Table: ensemble fitness (train/val/test) and rank among all ensembles."""
    try:
        import numpy as np
    except Exception:
        return ""

    minim = bool(gp.config['runcontrol'].get('minimisation', False))
    datasets = ['train', 'validation', 'test']
    rows_html = []
    any_present = False

    for ds in datasets:
        arr = _safe_get(gp.individuals, ['fitness','ensemble', ds])
        if arr is None:
            continue
        try:
            v = np.asarray(arr, dtype=float).reshape(-1)  # (num_ensembles,)
            val = float(v[ensemble_row])
            order = np.argsort(v) if minim else np.argsort(-v)
            rank = int(np.where(order == ensemble_row)[0][0]) + 1
            total = v.shape[0]
            rows_html.append(f"<tr><td>{ds}</td><td>{val:.6g}</td><td>{rank} / {total}</td></tr>")
            any_present = True
        except Exception:
            continue

    if not any_present:
        return ""

    return f"""
    <section class="ensemble-fitness">
      <h2>Ensemble fitness &amp; rank</h2>
      <div class="table-wrap tight">
        <table class="latex">
          <tr><th>Split</th><th>Fitness</th><th>Rank among ensembles</th></tr>
          {''.join(rows_html)}
        </table>
      </div>
    </section>
    """

################################################################################
# Weights tables (regression)
################################################################################

def _format_ensemble_weight_table_regress(ew, ensemble_row: int, num_pop: int, num_targets: int | None) -> str:
    """
    Ensemble mixing weights for regression, with explicit bias.

    Accepted shapes for ew[ensemble_row]:
      - (P+1,): [w_pop0, ..., w_pop{P-1}, bias]
      - (T, P+1): last column is bias per target
      - (P+1, T): last row is bias per target
      - (P,), (T, P), (P, T): no bias present (shown without bias column)

    Table layout:
      - Single target: one row labeled Target 0.
      - Multi-target: one row per target. Columns = weights for each population + a final “bias” column (when present).
    """
    try:
        import numpy as _np
        row = ew[ensemble_row]
        A = _np.asarray(row, dtype=float)
    except Exception:
        return ""

    def _fmt(x: float) -> str:
        return f"{float(x):.6g}"

    html_header = ""
    html_rows = ""

    # 1D: (P+1,) or (P,)
    if A.ndim == 1:
        if A.shape[0] == num_pop + 1:
            # weights + bias
            w = A[:num_pop].reshape(1, num_pop)
            b = A[-1:].reshape(1, 1)
            # Single target presentation
            cols = "".join(f"<th>w[pop={p}]</th>" for p in range(num_pop)) + "<th>bias</th>"
            html_header = f"<tr><th>Target</th>{cols}</tr>"
            cells = "".join(f"<td>{_fmt(w[0, p])}</td>" for p in range(num_pop)) + f"<td>{_fmt(b[0,0])}</td>"
            html_rows = f"<tr><td>0</td>{cells}</tr>"
        else:
            # No bias; (P,)
            w = A.reshape(1, -1)
            cols = "".join(f"<th>w[pop={p}]</th>" for p in range(w.shape[1]))
            html_header = f"<tr><th>Target</th>{cols}</tr>"
            cells = "".join(f"<td>{_fmt(w[0, p])}</td>" for p in range(w.shape[1]))
            html_rows = f"<tr><td>0</td>{cells}</tr>"

    # 2D: targets × (populations [+ bias]) OR (populations [+ bias]) × targets
    elif A.ndim == 2:
        r, c = A.shape
        # Case: (T, P+1) -> last column bias
        if c == num_pop + 1:
            T = r
            W = A[:, :num_pop]                   # (T, P)
            B = A[:, -1].reshape(T, 1)          # (T, 1)
            cols = "".join(f"<th>w[pop={p}]</th>" for p in range(num_pop)) + "<th>bias</th>"
            html_header = f"<tr><th>Target</th>{cols}</tr>"
            rows = []
            for t in range(T):
                cells = "".join(f"<td>{_fmt(W[t, p])}</td>" for p in range(num_pop)) + f"<td>{_fmt(B[t,0])}</td>"
                rows.append(f"<tr><td>{t}</td>{cells}</tr>")
            html_rows = "\n".join(rows)

        # Case: (P+1, T) -> last row bias
        elif r == num_pop + 1:
            T = c
            W = A[:num_pop, :].T                 # (T, P)
            B = A[-1, :].reshape(T, 1)          # (T, 1)
            cols = "".join(f"<th>w[pop={p}]</th>" for p in range(num_pop)) + "<th>bias</th>"
            html_header = f"<tr><th>Target</th>{cols}</tr>"
            rows = []
            for t in range(T):
                cells = "".join(f"<td>{_fmt(W[t, p])}</td>" for p in range(num_pop)) + f"<td>{_fmt(B[t,0])}</td>"
                rows.append(f"<tr><td>{t}</td>{cells}</tr>")
            html_rows = "\n".join(rows)

        # No bias present; try (T, P)
        elif c == num_pop:
            T = r
            W = A
            cols = "".join(f"<th>w[pop={p}]</th>" for p in range(num_pop))
            html_header = f"<tr><th>Target</th>{cols}</tr>"
            rows = []
            for t in range(T):
                cells = "".join(f"<td>{_fmt(W[t, p])}</td>" for p in range(num_pop))
                rows.append(f"<tr><td>{t}</td>{cells}</tr>")
            html_rows = "\n".join(rows)

        # No bias present; try (P, T)
        elif r == num_pop:
            T = c
            W = A.T                              # (T, P)
            cols = "".join(f"<th>w[pop={p}]</th>" for p in range(num_pop))
            html_header = f"<tr><th>Target</th>{cols}</tr>"
            rows = []
            for t in range(T):
                cells = "".join(f"<td>{_fmt(W[t, p])}</td>" for p in range(num_pop))
                rows.append(f"<tr><td>{t}</td>{cells}</tr>")
            html_rows = "\n".join(rows)

        else:
            # Unknown shape — don’t crash, show nothing
            return ""

    else:
        return ""

    return f"""
    <section class="ensemble-weights">
      <h2>Ensemble mixing weights</h2>
      <div class="table-wrap">
        <table class="latex">{html_header}{html_rows}</table>
      </div>
    </section>
    """


def _format_linreg_weights_table(WG, num_targets: int | None) -> Tuple[str, str]:
    """
    Linear regression head weights:
      - Prefer (T, G+1) with last column = bias.
      - Also supports (G+1, T) with last row = bias.
      - Also supports (G+1,) for single-target (last entry bias).
    Returns (table_html, yhat_equations_html).
    """
    if WG is None:
        return "", ""

    try:
        import numpy as _np
        A = _np.asarray(WG, dtype=float)
    except Exception:
        return "", ""

    # Normalize to (T, G) weights + (T,1) bias (or None)
    if A.ndim == 1:
        # (G+1,)
        if A.shape[0] >= 1:
            W = A[:-1].reshape(1, -1)
            b = A[-1:].reshape(1, 1)
        else:
            return "", ""
    elif A.ndim == 2:
        r, c = A.shape
        # If rows look like targets:
        # (T, G+1) -> last col bias
        # else (G+1, T) -> last row bias
        if c >= 1 and (num_targets is None or r == num_targets):
            if c >= 2:
                W = A[:, :-1]
                b = A[:, -1].reshape(-1, 1)
            else:
                W = A
                b = None
        else:
            # treat as transposed
            if r >= 2:
                W = A[:-1, :].T
                b = A[-1, :].reshape(-1, 1)
            else:
                W = A.T
                b = None
    else:
        return "", ""

    T, G = W.shape if W.ndim == 2 else (1, W.shape[0])

    # Equations
    lines = []
    for t in range(T):
        terms = [f"w[target={t}, g={j}]·G_j" for j in range(G)]
        if b is not None:
            terms.append("bias[target]")
        lines.append("ŷ[target={}] = ".format(t) + " + ".join(terms))
    eq_block = "<details><summary>Linear head (structure)</summary><pre>" + "\n".join(lines) + "</pre></details>"

    # Table
    header = "<tr><th>Target</th>" + "".join(f"<th>w[g={j}]</th>" for j in range(G)) + ("<th>bias</th>" if b is not None else "") + "</tr>"
    rows = []
    for t in range(T):
        cells = "".join(f"<td>{float(W[t, j]):.6g}</td>" for j in range(G))
        if b is not None:
            cells += f"<td>{float(b[t, 0]):.6g}</td>"
        rows.append(f"<tr><td>{t}</td>{cells}</tr>")
    table = f"""
    <details class="wgt">
      <summary>Linear regression weights (genes → target){' + bias' if b is not None else ''}</summary>
      <div class="table-wrap">
        <table class="latex">{header}{''.join(rows)}</table>
      </div>
    </details>
    """
    return table, eq_block

################################################################################
# HTML helpers
################################################################################

def _get_num_targets(gp) -> int | None:
    """Infer target dimensionality from ytrain if possible."""
    try:
        import numpy as np
        ytr = gp.userdata['ytrain']
        A = np.asarray(ytr)
        if A.ndim == 1:
            return 1
        if A.ndim == 2:
            return int(A.shape[1])
        return None
    except Exception:
        return None

def _header_block(title: str) -> str:
    return f"""
    <header>
      <h1>{html.escape(title)}</h1>
      <p>This report shows the chosen ensemble, its member individuals per population, and each member&apos;s gene trees.
      Node labels are the exact function names from the GenForge run (e.g., <code>plus</code>, <code>minus</code>, <code>times</code>, <code>divide</code>).
      For regression, we display the learned linear head weights (genes → target) with an explicit <em>bias</em> column when available, ensemble mixing weights, and fitness values (with ranks).</p>
    </header>
    """

def _safe_get(obj: Any, keys: List[str]):
    cur = obj
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur

def _wrap_html(title: str, header: str, ensemble_table: str, ensemble_fitness: str, members: str) -> str:
    # NOTE: double braces to keep CSS braces intact inside an f-string
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg:#f3f4f6;          /* page background (light gray) */
      --card:#eef0f3;        /* cards slightly darker than page */
      --tree:#e5e7eb;        /* tree area background */
      --muted:#6b7280;       /* text-muted */
      --text:#111827;        /* main text */
      --accent:#2563eb;      /* blue */
      --border:#d1d5db;      /* light border */
      --chip:#f3f4f6;        /* subtle chip */
      --latex-font:"Latin Modern Math","STIX Two Math","CMU Serif","Times New Roman",serif;
    }}
    html,body {{ background:var(--bg); color:var(--text); margin:0; font:14px/1.45 ui-sans-serif, system-ui, Segoe UI, Roboto, Arial; }}
    header {{ padding:28px 20px 8px 20px; border-bottom:1px solid var(--border); }}
    header h1 {{ margin:0 0 8px 0; font-size:22px; }}
    header p {{ margin:0; color:var(--muted); }}

    section.ensemble-weights, section.member, section.ensemble-fitness {{ padding:20px; border-bottom:1px solid var(--border); }}
    h2, h3, h4 {{ margin:0 0 12px 0; }}

    .table-wrap {{ overflow:auto; border:1px solid var(--border); border-radius:10px; background:var(--card); }}
    .table-wrap.tight table {{ min-width: 360px; }}
    table {{ border-collapse:collapse; width:100%; min-width:420px; }}
    th, td {{ padding:8px 10px; border-bottom:1px solid var(--border); text-align:right; color:var(--text); }}
    th:first-child, td:first-child {{ text-align:left; }}
    table.latex th, table.latex td {{ font-family: var(--latex-font); }}

    details summary {{ cursor:pointer; color:var(--accent); margin:12px 0; }}
    details.wgt summary {{ margin-top:18px; }}

    .gene-grid {{ display:grid; gap:14px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
    .gene-card {{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:12px; }}
    .gene-title {{ font-weight:600; margin-bottom:6px; }}
    .gene-code {{ color:var(--muted); margin-bottom:8px; }}
    /* Wrap long gene strings and avoid overlap with SVG */
    .gene-code code {{
      background:var(--chip);
      padding:3px 6px;
      border-radius:6px;
      display:block;
      white-space:pre-wrap;       /* allow wrapping & newlines */
      word-break:break-word;
      overflow-wrap:anywhere;
      line-height:1.35;
      margin-bottom:6px;
    }}

    .gene-svg {{ overflow:auto; background:var(--tree); border-radius:10px; }}

    .metrics h4 {{ margin: 0 0 8px 0; }}
  </style>
</head>
<body>
  {header}
  {ensemble_table}
  {ensemble_fitness}
  {members}
</body>
</html>"""

################################################################################
# (End of module)
################################################################################
