
"""
DWARF Architecture Tree Visualizer v2
Автоматически строит collapsible tree с иерархией классов,
файловыми структурами реализаций и функциями utils.

Использование:
    python generate_tree.py [путь_к_папке] [выходной_файл.html]
"""
import ast
import os
import sys
import json


def get_name(node):
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        parts = []
        n = node
        while isinstance(n, ast.Attribute):
            parts.append(n.attr)
            n = n.value
        if isinstance(n, ast.Name):
            parts.append(n.id)
        return ".".join(reversed(parts))
    return None


def get_func_signature(node):
    """Возвращает краткую сигнатуру функции: имя(арг1, арг2...)"""
    args = [a.arg for a in node.args.args]
    # Убираем self/cls для методов
    if args and args[0] in ("self", "cls"):
        args = args[1:]
    return f"{node.name}({', '.join(args)})" if args else f"{node.name}()"


def analyze_repo(root_dir):
    """Анализирует репозиторий: классы, функции, файлы"""
    classes = {}      # имя -> {file, bases, children, methods}
    files = {}        # rel_path -> {classes: [], functions: []}
    
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, fname)
            rel = os.path.relpath(fpath, root_dir).replace("\\", "/")
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    source = f.read()
                if not source.strip():
                    continue
                tree = ast.parse(source)
            except Exception:
                continue
            
            file_data = {"classes": [], "functions": []}
            
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    bases = [get_name(b) for b in node.bases if get_name(b)]
                    methods = []
                    for item in ast.iter_child_nodes(node):
                        if isinstance(item, ast.FunctionDef):
                            methods.append(get_func_signature(item))
                    
                    cls_info = {
                        "name": node.name,
                        "file": rel,
                        "bases": bases,
                        "children": [],
                        "methods": methods,
                    }
                    classes[node.name] = cls_info
                    file_data["classes"].append(cls_info)
                
                elif isinstance(node, ast.FunctionDef):
                    file_data["functions"].append(get_func_signature(node))
            
            files[rel] = file_data
    
    # Строим дерево наследования
    for name, info in classes.items():
        for base in info["bases"]:
            if base in classes and base != name:
                classes[base]["children"].append(name)
    
    return classes, files


def get_group(rel_path):
    lower = rel_path.lower()
    if "/attack_solutions/" in lower:
        return "attack"
    elif "/embedding_solutions/" in lower:
        return "embed"
    elif "/expertise_solutions/" in lower:
        return "expert"
    elif "/ds_solutions/" in lower:
        return "data"
    elif "/common_utils/" in lower or "/core/utils/" in lower or "/ready_solutions/utils/" in lower or "/logs/" in lower:
        return "utils"
    elif "/core/" in lower:
        return "core"
    elif "/tests/" in lower:
        return "test"
    else:
        return "other"


def make_class_node(cls_name, classes, visited=None):
    """Рекурсивно строит поддерево класса по наследованию"""
    if visited is None:
        visited = set()
    if cls_name in visited or cls_name not in classes:
        return None
    visited.add(cls_name)
    
    info = classes[cls_name]
    group = get_group(info["file"])
    
    node = {
        "name": cls_name,
        "type": "class",
        "group": group,
        "path": info["file"],
        "bases": info["bases"],
    }
    
    children = []
    for child_name in sorted(info["children"]):
        child_node = make_class_node(child_name, classes, visited.copy())
        if child_node:
            children.append(child_node)
    
    if children:
        node["children"] = children
    
    return node


def build_tree_data(classes, files, root_dir):
    """Строит иерархическое дерево для D3.js"""
    
    tree = {
        "name": "DWARF Framework",
        "type": "root",
        "group": "root",
        "children": []
    }
    
    # ============================================================
    #  1. ЯДРО — иерархия классов (метаклассы + ABC + категории)
    # ============================================================
    core_children = []
    
    # Метаклассы
    meta_names = ["Attack_Core_Meta", "Embedding_Core_Meta", "Expertise_Core_Meta", "Ds_Core_Meta"]
    meta_nodes = []
    for mc in meta_names:
        if mc in classes:
            meta_nodes.append(make_class_node(mc, classes))
    if meta_nodes:
        core_children.append({
            "name": "Метаклассы",
            "type": "group",
            "group": "core",
            "children": meta_nodes
        })
    
    # Базовые ABC
    abc_names = ["Attack_Core", "Embedding_Core", "Expertise_Core", "Ds_Core"]
    abc_nodes = []
    for abc in abc_names:
        if abc not in classes:
            continue
        node = make_class_node(abc, classes)
        if node:
            # Красивые эмодзи-префиксы
            if abc == "Attack_Core":
                node["name"] = abc
            elif abc == "Embedding_Core":
                node["name"] = abc
            elif abc == "Expertise_Core":
                node["name"] = abc
            elif abc == "Ds_Core":
                node["name"] = abc
            abc_nodes.append(node)
    
    if abc_nodes:
        core_children.append({
            "name": "Базовые ABC",
            "type": "group",
            "group": "core",
            "children": abc_nodes
        })
    
    if core_children:
        tree["children"].append({
            "name": "⚙️ Ядро",
            "type": "group",
            "group": "core",
            "children": core_children
        })
    
    # ============================================================
    #  2. РЕАЛИЗАЦИИ — файловая структура ready_solutions
    # ============================================================
    solutions_dirs = {
        "attack_solutions": ("Атаки", "attack"),
        "embedding_solutions": ("Встраивание", "embed"),
        "expertise_solutions": ("Экспертиза", "expert"),
        "ds_solutions": ("🟣 Датасеты", "data"),
    }
    
    solutions_children = []
    for dirname, (label, group) in solutions_dirs.items():
        dir_files = []
        for rel, fdata in sorted(files.items()):
            if dirname not in rel.replace("\\", "/"):
                continue
            # Пропускаем __init__.py без содержимого
            basename = os.path.basename(rel)
            if not fdata["classes"] and not fdata["functions"] and basename == "__init__.py":
                continue
            
            file_node = {
                "name": basename,
                "type": "file",
                "group": group,
                "path": rel,
            }
            file_children = []
            
            # Классы файла
            for cls in fdata["classes"]:
                cls_node = {
                    "name": cls["name"],
                    "type": "class",
                    "group": group,
                    "path": rel,
                    "bases": cls["bases"],
                }
                # Методы класса
                if cls["methods"]:
                    cls_node["children"] = [
                        {"name": m, "type": "method", "group": group, "path": rel}
                        for m in cls["methods"]
                    ]
                file_children.append(cls_node)
            
            # Функции файла (если есть)
            for func in fdata["functions"]:
                file_children.append({
                    "name": func,
                    "type": "function",
                    "group": group,
                    "path": rel,
                })
            
            if file_children:
                file_node["children"] = file_children
            dir_files.append(file_node)
        
        if dir_files:
            solutions_children.append({
                "name": label,
                "type": "group",
                "group": group,
                "children": dir_files
            })
    
    if solutions_children:
        tree["children"].append({
            "name": "Реализации",
            "type": "group",
            "group": "other",
            "children": solutions_children
        })
    
    # ============================================================
    #  3. УТИЛИТЫ — файлы с функциями
    # ============================================================
    util_files = []
    for rel, fdata in sorted(files.items()):
        group = get_group(rel)
        if group != "utils":
            continue
        basename = os.path.basename(rel)
        if not fdata["classes"] and not fdata["functions"] and basename == "__init__.py":
            continue
        
        file_node = {
            "name": basename,
            "type": "file",
            "group": "utils",
            "path": rel,
        }
        file_children = []
        
        for cls in fdata["classes"]:
            cls_node = {
                "name": cls["name"],
                "type": "class",
                "group": "utils",
                "path": rel,
                "bases": cls["bases"],
            }
            if cls["methods"]:
                cls_node["children"] = [
                    {"name": m, "type": "method", "group": "utils", "path": rel}
                    for m in cls["methods"]
                ]
            file_children.append(cls_node)
        
        for func in fdata["functions"]:
            file_children.append({
                "name": func,
                "type": "function",
                "group": "utils",
                "path": rel,
            })
        
        if file_children:
            file_node["children"] = file_children
        util_files.append(file_node)
    
    if util_files:
        tree["children"].append({
            "name": "🛠️ Утилиты",
            "type": "group",
            "group": "utils",
            "children": util_files
        })
    
    # ============================================================
    #  4. ТОЧКА ВХОДА
    # ============================================================
    entry_files = []
    for rel, fdata in sorted(files.items()):
        basename = os.path.basename(rel)
        if basename in ("main.py", "example.py"):
            file_node = {
                "name": basename,
                "type": "file",
                "group": "other",
                "path": rel,
            }
            file_children = []
            for cls in fdata["classes"]:
                file_children.append({
                    "name": cls["name"],
                    "type": "class",
                    "group": "other",
                    "path": rel,
                    "bases": cls["bases"],
                })
            for func in fdata["functions"]:
                file_children.append({
                    "name": func,
                    "type": "function",
                    "group": "other",
                    "path": rel,
                })
            if file_children:
                file_node["children"] = file_children
            entry_files.append(file_node)
    
    if entry_files:
        tree["children"].append({
            "name": "🚀 Точка входа",
            "type": "group",
            "group": "other",
            "children": entry_files
        })
    
    return tree


def generate_html(tree_data, output_path):
    data_json = json.dumps(tree_data, ensure_ascii=False)
    
    html = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DWARF Architecture - Tree</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:#0d1117;color:#c9d1d9;overflow:hidden;width:100vw;height:100vh}
svg{width:100%;height:100%;cursor:grab}
svg:active{cursor:grabbing}
.node circle{stroke-width:3px;cursor:pointer;transition:all .3s}
.node circle:hover{filter:drop-shadow(0 0 10px currentColor);transform:scale(1.15)}
.node.collapsed circle{stroke-dasharray:4,2}
.node text{font-size:13px;font-weight:500;fill:#f0f6fc;text-shadow:0 1px 4px rgba(0,0,0,.9);pointer-events:none}
.node.collapsed text{fill:#8b949e}
.node.root text{font-size:17px;font-weight:700}
.node.group text{font-size:15px;font-weight:600}
.node.file text{font-size:13px;font-weight:600}
.node.method text,.node.function text{font-size:11px;font-style:italic;fill:#8b949e}
.link{fill:none;stroke-opacity:.6;transition:all .3s}
.link.inheritance{stroke:#58a6ff;stroke-width:3px}
.link.contains{stroke:#30363d;stroke-width:1px;opacity:.25}
.link.highlighted{stroke-opacity:1;stroke-width:4px;filter:drop-shadow(0 0 4px currentColor)}
#controls{position:fixed;top:12px;left:12px;background:rgba(22,27,34,.95);border:1px solid #30363d;border-radius:10px;padding:14px;z-index:100;backdrop-filter:blur(8px);max-width:300px;box-shadow:0 8px 24px rgba(0,0,0,.4)}
#controls h1{font-size:16px;color:#f0f6fc;margin-bottom:6px}
#controls p{font-size:11px;color:#8b949e;margin-bottom:10px;line-height:1.4}
.legend-item{display:flex;align-items:center;gap:6px;font-size:11px;margin:4px 0;color:#c9d1d9}
.legend-dot{width:12px;height:12px;border-radius:50%;border:2px solid;flex-shrink:0}
.btn{background:#21262d;border:1px solid #30363d;color:#c9d1d9;padding:5px 10px;border-radius:5px;cursor:pointer;font-size:11px;margin-top:6px;margin-right:4px;transition:.2s}
.btn:hover{background:#30363d;border-color:#58a6ff}
#tooltip{position:absolute;background:rgba(22,27,34,.98);border:1px solid #30363d;border-radius:8px;padding:12px 16px;max-width:380px;font-size:12px;line-height:1.5;color:#c9d1d9;pointer-events:none;opacity:0;transition:opacity .2s;z-index:200;box-shadow:0 8px 24px rgba(0,0,0,.5)}
#tooltip .tt-title{font-size:14px;font-weight:600;color:#f0f6fc;margin-bottom:4px}
#tooltip .tt-meta{color:#8b949e;font-size:10px;margin-bottom:6px}
#tooltip .tt-bases{margin-top:6px;padding-top:6px;border-top:1px solid #30363d;color:#58a6ff}
#tooltip .tt-methods{margin-top:6px;padding-top:6px;border-top:1px solid #30363d;color:#8b949e;max-height:120px;overflow-y:auto}
#counter{position:fixed;bottom:12px;right:12px;background:rgba(22,27,34,.9);border:1px solid #30363d;border-radius:6px;padding:6px 12px;font-size:11px;color:#8b949e;z-index:100}
</style>
</head>
<body>
<svg id="tree-svg"></svg>
<div id="controls">
<h1>DWARF Architecture</h1>
<p>Дерево классов, реализаций и функций.<br><b>Клик</b> - свернуть/развернуть. <b>Drag</b> - панорама. <b>Колесо</b> - зум.</p>
<div style="border-top:1px solid #30363d;margin:8px 0;padding-top:8px">
<div class="legend-item"><div class="legend-dot" style="background:#1565c0;border-color:#42a5f5"></div><span>Ядро (core)</span></div>
<div class="legend-item"><div class="legend-dot" style="background:#c62828;border-color:#ff5252"></div><span>Атаки</span></div>
<div class="legend-item"><div class="legend-dot" style="background:#2e7d32;border-color:#66bb6a"></div><span>Встраивание</span></div>
<div class="legend-item"><div class="legend-dot" style="background:#ef6c00;border-color:#ff9800"></div><span>Экспертиза</span></div>
<div class="legend-item"><div class="legend-dot" style="background:#6a1b9a;border-color:#ab47bc"></div><span>Датасеты</span></div>
<div class="legend-item"><div class="legend-dot" style="background:#546e7a;border-color:#90a4ae"></div><span>Утилиты / Прочее</span></div>
</div>
<div style="border-top:1px solid #30363d;margin:8px 0;padding-top:8px">
<div class="legend-item"><span style="color:#58a6ff;font-weight:bold">━━▶</span><span>Наследование / Содержит</span></div>
</div>
<button class="btn" onclick="expandAll()">Развернуть всё</button>
<button class="btn" onclick="collapseAll()">Свернуть всё</button>
<button class="btn" onclick="resetZoom()">Сброс зума</button>
</div>
<div id="tooltip"></div>
<div id="counter"></div>
<script>
const treeData = """ + data_json + """;

const colors = {
  root: '#161b22', core: '#1565c0', attack: '#c62828', embed: '#2e7d32',
  expert: '#ef6c00', data: '#6a1b9a', utils: '#546e7a', test: '#795548', other: '#455a64'
};
const strokes = {
  root: '#58a6ff', core: '#42a5f5', attack: '#ff5252', embed: '#66bb6a',
  expert: '#ff9800', data: '#ab47bc', utils: '#90a4ae', test: '#a1887f', other: '#78909c'
};
const radiusMap = {root: 28, group: 18, file: 11, class: 8, method: 5, function: 5};

const margin = {top: 40, right: 220, bottom: 40, left: 120};
const width = window.innerWidth;
const height = window.innerHeight;
const svg = d3.select("#tree-svg");
const g = svg.append("g").attr("transform", `translate(${margin.left},${height/2})`);

const zoom = d3.zoom().scaleExtent([0.03, 4]).on("zoom", e => g.attr("transform", e.transform));
svg.call(zoom);
svg.call(zoom.transform, d3.zoomIdentity.translate(margin.left, height/2).scale(0.75));

const treeMap = d3.tree().nodeSize([38, 260]);
let i = 0;
const duration = 500;
let root;

root = d3.hierarchy(treeData, d => d.children);
root.x0 = height / 2;
root.y0 = 0;

function collapseLevel(d) {
  if (d.children) {
    if (d.depth >= 3) {
      d._children = d.children;
      d.children = null;
    } else {
      d.children.forEach(collapseLevel);
    }
  }
}
root.children.forEach(collapseLevel);

update(root);
updateCounter();

function update(source) {
  const treeData = treeMap(root);
  const nodes = treeData.descendants();
  const links = treeData.links();
  nodes.forEach(d => { d.y = d.depth * 280; });

  const node = g.selectAll('g.node').data(nodes, d => d.id || (d.id = ++i));

  const nodeEnter = node.enter().append('g')
    .attr('class', d => 'node ' + (d.data.type || 'class') + (d._children ? ' collapsed' : ''))
    .attr("transform", d => `translate(${source.y0},${source.x0})`)
    .on('click', click)
    .on('mouseover', showTooltip)
    .on('mouseout', hideTooltip);

  nodeEnter.append('circle')
    .attr('r', 1e-6)
    .style("fill", d => colors[d.data.group || 'other'] || colors.other)
    .style("stroke", d => strokes[d.data.group || 'other'] || strokes.other);

  nodeEnter.append('text')
    .attr("dy", ".35em")
    .attr("x", d => (radiusMap[d.data.type] || 8) + 10)
    .attr("text-anchor", "start")
    .text(d => d.data.name)
    .style("fill-opacity", 1e-6);

  const nodeUpdate = node.merge(nodeEnter).transition().duration(duration)
    .attr("transform", d => `translate(${d.y},${d.x})`);

  nodeUpdate.select('circle')
    .attr('r', d => radiusMap[d.data.type] || 8)
    .style("fill", d => colors[d.data.group || 'other'] || colors.other)
    .style("stroke", d => strokes[d.data.group || 'other'] || strokes.other);

  nodeUpdate.select('text').style("fill-opacity", 1);
  nodeUpdate.attr('class', d => 'node ' + (d.data.type || 'class') + (d._children ? ' collapsed' : ''));

  const nodeExit = node.exit().transition().duration(duration)
    .attr("transform", d => `translate(${source.y},${source.x})`)
    .remove();
  nodeExit.select('circle').attr('r', 1e-6);
  nodeExit.select('text').style("fill-opacity", 1e-6);

  const link = g.selectAll('path.link').data(links, d => d.target.id);

  const linkEnter = link.enter().insert('path', "g")
    .attr("class", "link inheritance")
    .attr('d', d => {
      const o = {x: source.x0, y: source.y0};
      return diagonal(o, o);
    });

  const linkUpdate = link.merge(linkEnter).transition().duration(duration)
    .attr('d', d => diagonal(d.source, d.target));

  link.exit().transition().duration(duration)
    .attr('d', d => {
      const o = {x: source.x, y: source.y};
      return diagonal(o, o);
    })
    .remove();

  nodes.forEach(d => { d.x0 = d.x; d.y0 = d.y; });
}

function diagonal(s, d) {
  return `M ${s.y} ${s.x} C ${(s.y + d.y) / 2} ${s.x}, ${(s.y + d.y) / 2} ${d.x}, ${d.y} ${d.x}`;
}

function click(e, d) {
  if (d.children) {
    d._children = d.children;
    d.children = null;
  } else {
    d.children = d._children;
    d._children = null;
  }
  update(d);
  updateCounter();
}

function showTooltip(e, d) {
  const t = document.getElementById('tooltip');
  const data = d.data;
  const color = strokes[data.group || 'other'] || strokes.other;
  
  let typeLabel = data.type === 'root' ? 'Проект' : 
    (data.type === 'group' ? 'Группа' : 
    (data.type === 'file' ? 'Файл' : 
    (data.type === 'class' ? 'Класс' : 
    (data.type === 'method' ? 'Метод' : 'Функция'))));
  
  let basesHtml = '';
  if (data.bases && data.bases.length > 0) {
    basesHtml = `<div class="tt-bases">Наследуется от: ${data.bases.join(', ')}</div>`;
  }
  
  let methodsHtml = '';
  if (data.children && data.type === 'class') {
    const methods = data.children.filter(c => c.type === 'method' || c.type === 'function').map(c => c.name);
    if (methods.length > 0) {
      methodsHtml = `<div class="tt-methods">Методы/функции:<br/>${methods.slice(0, 20).join('<br/>')}${methods.length > 20 ? '<br/>... и ещё ' + (methods.length - 20) : ''}</div>`;
    }
  }
  
  t.innerHTML = `<div class="tt-title">${data.name}</div><div class="tt-meta">${typeLabel} · ${data.group || 'other'}${data.path ? ' · ' + data.path : ''}</div>${basesHtml}${methodsHtml}`;
  
  let left = e.pageX + 20;
  let top = e.pageY - 20;
  if (left + 400 > window.innerWidth) left = e.pageX - 400;
  if (top + 200 > window.innerHeight) top = e.pageY - 200;
  t.style.left = left + 'px';
  t.style.top = top + 'px';
  t.style.opacity = '1';

  g.selectAll('path.link').classed('highlighted', false);
  let curr = d;
  while (curr.parent) {
    g.selectAll('path.link').filter(l => l.target === curr).classed('highlighted', true);
    curr = curr.parent;
  }
}

function hideTooltip() {
  document.getElementById('tooltip').style.opacity = '0';
  g.selectAll('path.link').classed('highlighted', false);
}

function expandAll() {
  function expand(d) {
    if (d._children) { d.children = d._children; d._children = null; }
    if (d.children) d.children.forEach(expand);
  }
  expand(root);
  update(root);
  updateCounter();
}

function collapseAll() {
  function collapse(d) {
    if (d.children) { d._children = d.children; d.children = null; }
    if (d._children) d._children.forEach(collapse);
  }
  if (root.children) root.children.forEach(collapse);
  update(root);
  updateCounter();
}

function resetZoom() {
  svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity.translate(margin.left, height/2).scale(0.75));
}

function updateCounter() {
  const visible = root.descendants().length;
  document.getElementById('counter').textContent = `Узлов: ${visible}`;
}

window.addEventListener('resize', () => {
  svg.attr('width', window.innerWidth).attr('height', window.innerHeight);
});
</script>
</body>
</html>"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Сохранено: {os.path.abspath(output_path)}")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    output = sys.argv[2] if len(sys.argv) > 2 else "dwarf_tree.html"
    
    if not os.path.isdir(root):
        print(f"Ошибка: {root} не является папкой")
        sys.exit(1)
    
    print(f"Анализируем: {os.path.abspath(root)}")
    classes, files = analyze_repo(root)
    tree = build_tree_data(classes, files, root)
    
    total_classes = len(classes)
    total_funcs = sum(len(f["functions"]) for f in files.values())
    print(f"  Классов: {total_classes}")
    print(f"  Функций: {total_funcs}")
    
    generate_html(tree, output)
    print("Готово! Откройте файл в браузере.")


if __name__ == "__main__":
    main()
