"""[PACKAGE: pathlib / base64]
WHY: Comparing 2D lead structure depictions across 18+ de novo RL runs requires a unified,
     zero-friction visual comparison dashboard without opening dozens of editor tabs.
WHAT: Generates a self-contained HTML gallery with side-by-side comparison, instant thumbnail grid,
     and keyboard navigation (arrow keys) directly inside VS Code Simple Browser or standard browsers.
HOW: Scans _david/output for lead_structures_2d.png, embeds relative links/base64, extracts run metadata,
     and constructs an interactive split-view UI.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path


def build_gallery(output_dir: Path | str | None = None) -> Path:
    out_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent.parent / "output"
    images = sorted(out_dir.glob("**/lead_structures_2d.png"))
    diverse_images = sorted(out_dir.glob("**/diverse_lead_structures_2d.png"))
    all_images = sorted(set(images + diverse_images))

    records = []
    for p in all_images:
        rel = p.relative_to(out_dir)
        parts = rel.parts
        # Categorize run
        category = parts[0] if len(parts) > 2 else "Recent Runs"
        run_name = parts[1] if len(parts) > 2 else parts[0]
        sub_type = "Diverse Leads" if "diverse" in p.name else "Top Leads"

        # Read image as base64 for portable offline viewing
        with open(p, "rb") as img_f:
            b64 = base64.b64encode(img_f.read()).decode("ascii")

        records.append({
            "category": category,
            "run_name": run_name,
            "type": sub_type,
            "path": str(rel),
            "b64": b64,
        })

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DrugEx Lead Structures 2D Comparison Gallery</title>
    <style>
        :root {{
            --bg: #1e1e1e;
            --card-bg: #252526;
            --border: #3c3c3c;
            --accent: #007acc;
            --text: #cccccc;
            --text-bright: #ffffff;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
        }}
        header {{
            background: #2d2d2d;
            border-bottom: 1px solid var(--border);
            padding: 10px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-shrink: 0;
        }}
        h1 {{ font-size: 16px; font-weight: 600; color: var(--text-bright); }}
        .controls {{ display: flex; gap: 12px; align-items: center; font-size: 13px; }}
        .btn {{
            background: #3a3d41;
            color: var(--text-bright);
            border: 1px solid var(--border);
            padding: 5px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }}
        .btn.active {{ background: var(--accent); border-color: var(--accent); }}
        .main-container {{
            display: flex;
            flex: 1;
            overflow: hidden;
        }}
        /* Sidebar List */
        .sidebar {{
            width: 320px;
            background: #252526;
            border-right: 1px solid var(--border);
            overflow-y: auto;
            flex-shrink: 0;
            display: flex;
            flex-direction: column;
        }}
        .search-box {{
            padding: 8px 12px;
            border-bottom: 1px solid var(--border);
            position: sticky;
            top: 0;
            background: #252526;
            z-index: 10;
        }}
        .search-box input {{
            width: 100%;
            background: #3c3c3c;
            border: 1px solid #555;
            color: #fff;
            padding: 6px 10px;
            border-radius: 3px;
            font-size: 12px;
        }}
        .run-list {{ list-style: none; padding: 4px 0; }}
        .run-item {{
            padding: 8px 12px;
            cursor: pointer;
            border-bottom: 1px solid #2d2d2d;
            font-size: 12px;
            transition: background 0.15s;
        }}
        .run-item:hover {{ background: #2a2d2e; }}
        .run-item.active {{ background: #094771; color: #fff; font-weight: 600; }}
        .run-cat {{ font-size: 10px; text-transform: uppercase; color: #888; margin-bottom: 2px; }}
        .run-type-badge {{
            display: inline-block;
            font-size: 10px;
            padding: 1px 5px;
            border-radius: 3px;
            background: #333;
            color: #aaa;
            margin-top: 4px;
        }}
        /* Viewer Area */
        .viewer {{
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: auto;
            padding: 20px;
            align-items: center;
            justify-content: center;
        }}
        .image-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 16px;
            max-width: 95%;
            max-height: 95%;
            display: flex;
            flex-direction: column;
            align-items: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        }}
        .image-card img {{
            max-width: 100%;
            max-height: calc(100vh - 160px);
            object-fit: contain;
            border-radius: 4px;
            background: #ffffff;
            padding: 8px;
        }}
        .image-meta {{
            margin-top: 12px;
            font-size: 13px;
            color: var(--text-bright);
            text-align: center;
        }}
        /* Split Compare Mode */
        .split-view {{
            display: none;
            flex: 1;
            gap: 16px;
            overflow: hidden;
            padding: 16px;
        }}
        .split-pane {{
            flex: 1;
            display: flex;
            flex-direction: column;
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 12px;
            overflow: hidden;
        }}
        .split-pane select {{
            background: #3c3c3c;
            color: #fff;
            padding: 6px;
            border: 1px solid #555;
            border-radius: 3px;
            margin-bottom: 10px;
            font-size: 12px;
            width: 100%;
        }}
        .split-pane-img {{
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            background: #fff;
            border-radius: 4px;
            padding: 8px;
        }}
        .split-pane-img img {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }}
        .grid-view {{
            display: none;
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
            gap: 20px;
        }}
        .grid-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 12px;
            cursor: pointer;
            transition: transform 0.15s, border-color 0.15s;
        }}
        .grid-card:hover {{
            transform: translateY(-2px);
            border-color: var(--accent);
        }}
        .grid-card img {{
            width: 100%;
            height: 280px;
            object-fit: contain;
            background: #fff;
            border-radius: 4px;
            padding: 4px;
        }}
        .grid-card-title {{
            font-size: 12px;
            font-weight: 600;
            color: #fff;
            margin-top: 8px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
    </style>
</head>
<body>
    <header>
        <h1>🔬 DrugEx Lead Structures 2D Gallery ({len(records)} runs)</h1>
        <div class="controls">
            <button class="btn active" id="btnSingle" onclick="setViewMode('single')">Single & Arrow Nav</button>
            <button class="btn" id="btnSplit" onclick="setViewMode('split')">Side-by-Side Compare</button>
            <button class="btn" id="btnGrid" onclick="setViewMode('grid')">All Grid ({len(records)})</button>
        </div>
    </header>

    <div class="main-container">
        <!-- Sidebar Navigation -->
        <div class="sidebar" id="sidebar">
            <div class="search-box">
                <input type="text" id="filterInput" placeholder="Filter runs (e.g. e0.1, DNMT1, SA)..." oninput="filterRuns()">
            </div>
            <ul class="run-list" id="runList"></ul>
        </div>

        <!-- Single View -->
        <div class="viewer" id="singleViewer">
            <div class="image-card">
                <img id="singleImg" src="" alt="Lead Structures">
                <div class="image-meta" id="singleMeta"></div>
            </div>
        </div>

        <!-- Split Side-by-Side Compare View -->
        <div class="split-view" id="splitViewer">
            <div class="split-pane">
                <select id="leftSelect" onchange="updateSplit('left')"></select>
                <div class="split-pane-img"><img id="leftImg" src=""></div>
            </div>
            <div class="split-pane">
                <select id="rightSelect" onchange="updateSplit('right')"></select>
                <div class="split-pane-img"><img id="rightImg" src=""></div>
            </div>
        </div>

        <!-- Grid View -->
        <div class="grid-view" id="gridViewer"></div>
    </div>

    <script>
        const data = {html.escape(str(records)).replace("'", '"')};
        let currentIndex = 0;
        let viewMode = 'single';

        function init() {{
            renderList();
            renderSelects();
            renderGrid();
            selectRun(0);

            window.addEventListener('keydown', (e) => {{
                if (viewMode === 'single') {{
                    if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {{
                        selectRun((currentIndex + 1) % data.length);
                    }} else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {{
                        selectRun((currentIndex - 1 + data.length) % data.length);
                    }}
                }}
            }});
        }}

        function renderList() {{
            const list = document.getElementById('runList');
            list.innerHTML = data.map((r, i) => `
                <li class="run-item" id="item-${{i}}" onclick="selectRun(${{i}})">
                    <div class="run-cat">${{r.category}}</div>
                    <div>${{r.run_name}}</div>
                    <span class="run-type-badge">${{r.type}}</span>
                </li>
            `).join('');
        }}

        function renderSelects() {{
            const opts = data.map((r, i) => `<option value="${{i}}">${{r.category}} — ${{r.run_name}} (${{r.type}})</option>`).join('');
            document.getElementById('leftSelect').innerHTML = opts;
            document.getElementById('rightSelect').innerHTML = opts;
            if (data.length > 1) document.getElementById('rightSelect').selectedIndex = 1;
            updateSplit('left');
            updateSplit('right');
        }}

        function renderGrid() {{
            const grid = document.getElementById('gridViewer');
            grid.innerHTML = data.map((r, i) => `
                <div class="grid-card" onclick="selectRun(${{i}}); setViewMode('single');">
                    <img src="data:image/png;base64,${{r.b64}}" loading="lazy">
                    <div class="grid-card-title">${{r.run_name}}</div>
                    <span class="run-type-badge">${{r.category}}</span>
                </div>
            `).join('');
        }}

        function selectRun(idx) {{
            currentIndex = idx;
            document.querySelectorAll('.run-item').forEach((el, i) => {{
                el.classList.toggle('active', i === idx);
            }});
            const item = document.getElementById('item-' + idx);
            if (item) item.scrollIntoView({{ block: 'nearest' }});

            const r = data[idx];
            document.getElementById('singleImg').src = 'data:image/png;base64,' + r.b64;
            document.getElementById('singleMeta').innerHTML = `<strong>${{r.category}}</strong> / ${{r.run_name}} <span class="run-type-badge">${{r.type}}</span><br><small style="color:#888;">Use ↑ / ↓ arrow keys to flip runs</small>`;
        }}

        function updateSplit(side) {{
            const idx = document.getElementById(side + 'Select').value;
            const r = data[idx];
            document.getElementById(side + 'Img').src = 'data:image/png;base64,' + r.b64;
        }}

        function setViewMode(mode) {{
            viewMode = mode;
            document.getElementById('btnSingle').classList.toggle('active', mode === 'single');
            document.getElementById('btnSplit').classList.toggle('active', mode === 'split');
            document.getElementById('btnGrid').classList.toggle('active', mode === 'grid');

            document.getElementById('sidebar').style.display = mode === 'single' ? 'flex' : 'none';
            document.getElementById('singleViewer').style.display = mode === 'single' ? 'flex' : 'none';
            document.getElementById('splitViewer').style.display = mode === 'split' ? 'flex' : 'none';
            document.getElementById('gridViewer').style.display = mode === 'grid' ? 'grid' : 'none';
        }}

        function filterRuns() {{
            const q = document.getElementById('filterInput').value.toLowerCase();
            document.querySelectorAll('.run-item').forEach((el, i) => {{
                const r = data[i];
                const match = r.run_name.toLowerCase().includes(q) || r.category.toLowerCase().includes(q);
                el.style.display = match ? 'block' : 'none';
            }});
        }}

        window.onload = init;
    </script>
</body>
</html>
"""
    gallery_path = out_dir / "lead_structures_gallery.html"
    gallery_path.write_text(html_content, encoding="utf-8")
    print(f"Gallery successfully created at: {gallery_path} ({gallery_path.stat().st_size} bytes)")
    return gallery_path


if __name__ == "__main__":
    build_gallery()
