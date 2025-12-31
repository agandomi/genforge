# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 Mohammad Sadegh Khorshidi

import os
import html
from typing import List, Tuple, Any

################################################################################
# Public API
################################################################################

def write_ensemble_html(gp,
                        ensemble_row: int,
                        out_path: str,
                        title: str = "GenForge Ensemble Report") -> str:
    """
    Create a self-contained HTML file visualizing the chosen ensemble.

    Parameters
    ----------
    gp : GenForge GP object (already evaluated)
        Must expose:
          - gp.population[pop][ind] -> List[str] of gene strings (e.g., "plus(x1,times(x2,x3))")
          - gp.config['runcontrol']['num_pop'], gp.config['runcontrol']['minimisation']
          - gp.individuals['ensemble_idx'][ensemble_row, pop] -> int index of selected individual
          - (optional) gp.individuals['ensemble_weight'][ensemble_row] -> (num_pop,) or (num_pop, num_class)
          - (optional) gp.individuals['weight_genes'][pop][ind] -> 2D array (num_class, num_genes+1) [last col = bias]
          - (optional) gp.individuals['fitness']['isolated'][train|validation|test] -> (num_ind, num_pop)
          - (optional) gp.individuals['fitness']['ensemble'][train|validation|test] -> (num_ensembles,)

    ensemble_row : int
        Row in `gp.individuals['ensemble_idx']` that corresponds to the chosen ensemble.

    out_path : str
        Path to the HTML file to be written.

    title : str
        Report title (displayed at top of the HTML).

    Returns
    -------
    str : The absolute path to the written HTML.
    """
    # --- Gather ensemble membership -------------------------------------------------------
    num_pop = int(gp.config['runcontrol']['num_pop'])
    ens_idx = _safe_get(gp.individuals, ['ensemble_idx'])
    if ens_idx is None:
        raise RuntimeError("gp.individuals['ensemble_idx'] not found. Run the ensemble evaluation first.")
    if ensemble_row < 0 or ensemble_row >= ens_idx.shape[0]:
        raise IndexError(f"ensemble_row {ensemble_row} out of range [0, {ens_idx.shape[0]-1}]")

    selected_by_pop = []
    for p in range(num_pop):
        ind_idx = int(ens_idx[ensemble_row, p])
        selected_by_pop.append((p, ind_idx))

    # Optional data
    ensemble_weight = _safe_get(gp.individuals, ['ensemble_weight'])
    weight_genes = _safe_get(gp.individuals, ['weight_genes'])

    # --- Build sections for each (pop, ind) ----------------------------------------------
    member_sections = []
    max_num_class = _get_num_class(gp)

    for p, ind in selected_by_pop:
        # gene strings
        genes = list(gp.population[p][ind])  # list[str]
        ind_title = f"Population {p} · Individual {ind}"

        # learned weights (genes -> classes) + bias as last column
        WG = None
        try:
            WG = weight_genes[p][ind]
        except Exception:
            WG = None

        weights_table_html, z_equations_html = _format_weights_table(WG, max_num_class)

        # Render all gene trees for this individual
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

        # Isolated fitness + ranking block for this member
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
          {z_equations_html}
          {weights_table_html}
        </section>
        """)

    # --- Ensemble weights table (if available) -------------------------------------------
    ensemble_table_html = ""
    if ensemble_weight is not None:
        ensemble_table_html = _format_ensemble_weight_table(ensemble_weight, ensemble_row, num_pop, max_num_class)

    # --- Ensemble fitness + ranking -------------------------------------------------------
    ensemble_fit_html = _format_ensemble_fitness_block(gp, ensemble_row)

    # --- Compose full HTML ----------------------------------------------------------------
    html_text = _wrap_html(
        title=title,
        header=_header_block(title),
        ensemble_table=ensemble_table_html,
        ensemble_fitness=ensemble_fit_html,
        members="".join(member_sections),
    )

    # --- Write file -----------------------------------------------------------------------
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
    Parse strings like:
        'plus(x1,times(x2,[3.0]))'
        'times(x1,x2)'
        'x1'
        '[1.23]'
        'rand()'   # arity-0 function
    into a Node tree where labels are the exact function/terminal names.

    Grammar (informal):
        expr := func | var | const
        func := NAME '(' [expr (',' expr)*] ')'
        var  := 'x' DIGITS+
        const:= '[' any-non-']' ']'
    """
    s = s.strip()

    def parse_expr(i: int) -> Tuple[Node, int]:
        # constant like [1.23]
        if i < len(s) and s[i] == '[':
            j = i + 1
            while j < len(s) and s[j] != ']':
                j += 1
            if j >= len(s):
                raise ValueError("Unclosed constant bracket '['")
            # Keep the bracketed token as label in the AST;
            # rendering will strip the brackets visually.
            val = s[i:j+1]
            return Node(val, []), j + 1

        # variable like x12
        if i < len(s) and s[i] == 'x':
            j = i + 1
            while j < len(s) and s[j].isdigit():
                j += 1
            return Node(s[i:j], []), j

        # function name
        j = i
        while j < len(s) and (s[j].isalnum() or s[j] == '_' or s[j] == '$'):
            j += 1
        name = s[i:j]
        if j >= len(s) or s[j] != '(':
            # bare token fallback
            return Node(name, []), j
        # parse args in parentheses
        j += 1  # skip '('
        args = []
        if j < len(s) and s[j] == ')':
            # arity-0 function: name()
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
    """Return pixel width required by subtree."""
    if not node.children:
        return 2 * node_radius
    widths = [_measure(ch, node_radius, hgap) for ch in node.children]
    return max(2 * node_radius, sum(widths) + hgap * (len(widths) - 1))

def _layout(node: Node, x: float, y: float, node_radius=16, hgap=20, vgap=60):
    """
    Compute positions for each node (center x,y). Returns (nodes, edges) where:
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
    Render the tree as an inline SVG.
    Visual theme:
      - page bg:      #f3f4f6 (set in CSS)
      - tree bg:      #e5e7eb (slightly darker)
      - edges:        black
      - nodes:        light blue fill, black outline; labels black
    """
    # Layout
    nodes, edges, width = _layout(node, x=0, y=0, node_radius=node_radius, hgap=hgap, vgap=vgap)

    # Normalize to positive coordinates + margin
    xs = [x for _, x, _ in nodes] + [x1 for x1,_,x2,_ in edges] + [x2 for _,_,x2,_ in edges]
    ys = [y for _, _, y in nodes] + [y1 for _,y1,_,_ in edges] + [y2 for _,_,_,y2 in edges]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    pad = 24
    shift_x = -minx + pad
    shift_y = -miny + pad
    W = int(maxx - minx + 2 * pad)
    H = int(maxy - miny + 2 * pad)

    # Colors
    svg_bg = "#e5e7eb"      # tree area background
    edge_color = "#000000"  # black
    node_fill = "#cfe8ff"   # light blue
    node_stroke = "#000000" # black outline
    text_fill = "#000000"   # black labels

    # Build SVG
    out = []
    out.append(f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Gene tree">')
    out.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="{svg_bg}"/>')
    # edges
    for (x1, y1, x2, y2) in edges:
        out.append(f'<line x1="{x1+shift_x:.1f}" y1="{y1+shift_y:.1f}" x2="{x2+shift_x:.1f}" y2="{y2+shift_y:.1f}" stroke="{edge_color}" stroke-width="1.5"/>')
    # nodes
    for (label, x, y) in nodes:
        cx, cy = x + shift_x, y + shift_y
        # strip brackets for constant nodes like "[1.23]"
        lab = str(label)
        if len(lab) >= 2 and lab[0] == "[" and lab[-1] == "]":
            lab = lab[1:-1]
        safe = html.escape(lab)
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{node_radius}" fill="{node_fill}" stroke="{node_stroke}" stroke-width="1.2"/>')
        out.append(f'<text x="{cx:.1f}" y="{cy+4:.1f}" font-size="11" text-anchor="middle" fill="{text_fill}" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial">{safe}</text>')
    out.append("</svg>")
    return "\n".join(out)

################################################################################
# Fitness/Ranks helpers
################################################################################

def _format_member_fitness_block(gp, pop_idx: int, ind_idx: int) -> str:
    """Small table showing train/val/test fitness and rank of an isolated individual among its population."""
    try:
        import numpy as np
    except Exception:
        return ""

    minim = bool(gp.config['runcontrol'].get('minimisation', False))
    datasets = ['train', 'validation', 'test']
    rows_html = []
    any_present = False

    # Pop size (columns are populations)
    iso_train = _safe_get(gp.individuals, ['fitness','isolated','train'])
    pop_size = None
    try:
        A = np.asarray(iso_train)
        pop_size = int(A.shape[0])  # rows are individuals
    except Exception:
        pass

    for ds in datasets:
        arr = _safe_get(gp.individuals, ['fitness','isolated', ds])
        if arr is None:
            continue
        try:
            M = np.asarray(arr, dtype=float)  # shape (num_ind, num_pop)
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
    """Table showing ensemble fitness (train/val/test) and rank among all ensembles."""
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
# HTML helpers
################################################################################

def _header_block(title: str) -> str:
    return f"""
    <header>
      <h1>{html.escape(title)}</h1>
      <p>This report shows the chosen ensemble, the member individuals per population, and each member&apos;s gene trees.
      Node labels are the exact function names from the GenForge run (e.g., <code>plus</code>, <code>minus</code>, <code>times</code>, <code>divide</code>). When available, softmax weights (genes → classes) show an explicit final <em>bias</em> column; ensemble mixing weights and fitness values (with ranks) are also shown.</p>
    </header>
    """

def _format_ensemble_weight_table(ew, ensemble_row: int, num_pop: int, num_class: int) -> str:
    # Accept several shapes: (P,), (P,C), or (C,P)
    try:
        row = ew[ensemble_row]
    except Exception:
        return ""

    try:
        import numpy as _np
        A = _np.asarray(row, dtype=float)
    except Exception:
        return ""

    if A.ndim == 1:
        # One weight per population (class-agnostic)
        header = "<tr><th>Population</th><th>Weight</th></tr>"
        body = "\n".join(f"<tr><td>{p}</td><td>{float(A[p]):.6g}</td></tr>" for p in range(min(num_pop, A.shape[0])))
    elif A.ndim == 2:
        # Ensure populations are rows
        if A.shape[0] == num_pop:
            M = A
        elif A.shape[1] == num_pop:
            M = A.T
        else:
            M = A
        C = M.shape[1] if num_class is None else num_class
        if C != M.shape[1] and C is not None and C <= M.shape[1]:
            M = M[:, :C]
        header = "<tr><th>Population</th>" + "".join(f"<th>w[c={c}]</th>" for c in range(M.shape[1])) + "</tr>"
        rows = []
        for p in range(min(num_pop, M.shape[0])):
            cells = "".join(f"<td>{float(M[p, c]):.6g}</td>" for c in range(M.shape[1]))
            rows.append(f"<tr><td>{p}</td>{cells}</tr>")
        body = "\n".join(rows)
    else:
        return ""

    return f"""
    <section class="ensemble-weights">
      <h2>Ensemble mixing weights</h2>
      <div class="table-wrap">
        <table class="latex">{header}{body}</table>
      </div>
    </section>
    """

def _format_weights_table(WG, num_class: int) -> Tuple[str, str]:
    """
    WG expected 2D array:
      rows = num_class, columns = num_genes + 1 (last column = bias).
    Also supports transposed variant (num_genes + 1, num_class) with last row = bias.
    Returns (table_html, z_equations_html).
    """
    if WG is None:
        return "", ""

    try:
        import numpy as _np
        A = _np.asarray(WG, dtype=float)
    except Exception:
        return "", ""

    if A.ndim != 2:
        return "", ""

    # Determine orientation and split weights/bias
    C_decl = num_class
    r, c = A.shape
    W = None
    bias = None

    if C_decl is not None and r == C_decl and c >= 1:
        # class-major; assume last column is bias if available
        if c >= 2:
            W = A[:, :-1]
            bias = A[:, -1].reshape(-1, 1)
        else:
            W = A
    elif C_decl is not None and c == C_decl and r >= 1:
        # transposed class-major; assume last row is bias if available
        if r >= 2:
            W = A[:-1, :].T  # (C, G)
            bias = A[-1, :].reshape(-1, 1)
        else:
            W = A.T
    else:
        # fallback: prefer class-major interpretation
        if r <= c:
            if c >= 2:
                W = A[:, :-1]
                bias = A[:, -1].reshape(-1, 1)
            else:
                W = A
        else:
            if r >= 2:
                W = A[:-1, :].T
                bias = A[-1, :].reshape(-1, 1)
            else:
                W = A.T

    C, G = W.shape
    # z-equations per class (symbolic skeleton)
    z_lines = []
    for cidx in range(C):
        terms = [f"w[c={cidx}, g={j}]·G_j" for j in range(G)]
        if bias is not None:
            terms.append("bias[c]")
        z_lines.append(f"z[c={cidx}] = " + " + ".join(terms))
    z_block = "<details><summary>Softmax logits per class (structure)</summary><pre>" + "\n".join(z_lines) + "</pre></details>"

    # weights table
    header = "<tr><th>Class</th>" + "".join(f"<th>w[g={j}]</th>" for j in range(G)) + ("<th>bias</th>" if bias is not None else "") + "</tr>"
    rows = []
    for cidx in range(C):
        cells = "".join(f"<td>{float(W[cidx, j]):.6g}</td>" for j in range(G))
        if bias is not None:
            cells += f"<td>{float(bias[cidx, 0]):.6g}</td>"
        rows.append(f"<tr><td>{cidx}</td>{cells}</tr>")
    table = f"""
    <details class="wgt">
      <summary>Softmax weights (genes → classes){' + bias' if bias is not None else ''}</summary>
      <div class="table-wrap">
        <table class="latex">{header}{''.join(rows)}</table>
      </div>
    </details>
    """
    return table, z_block

def _get_num_class(gp) -> int:
    # Prefer runcontrol, then userdata; last resort: None
    try:
        return int(gp.config['runcontrol']['num_class'])
    except Exception:
        pass
    try:
        return int(gp.userdata.get('num_class'))
    except Exception:
        return None

def _safe_get(obj: Any, keys: List[str]):
    cur = obj
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur

def _wrap_html(title: str, header: str, ensemble_table: str, ensemble_fitness: str, members: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      /* Light theme */
      --bg:#f3f4f6;          /* page background (light gray) */
      --card:#eef0f3;        /* cards slightly darker than page */
      --tree:#e5e7eb;        /* tree area background a bit darker than page */
      --muted:#6b7280;       /* text-muted */
      --text:#111827;        /* main text (near-black) */
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
    /* Wrap long gene strings onto multiple lines and keep them away from the SVG */
    .gene-code code {{
      background:var(--chip);
      padding:3px 6px;
      border-radius:6px;
      display:block;
      white-space:pre-wrap;       /* allow newlines and wrapping */
      word-break:break-word;      /* break long tokens if needed */
      overflow-wrap:anywhere;     /* aggressive wrapping for very long strings */
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
