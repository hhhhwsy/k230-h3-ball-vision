from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import shutil
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


APP_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>小铁球 YOLO 标注复核</title>
  <style>
    :root{color-scheme:dark;--bg:#11151b;--panel:#1a2029;--line:#343e4c;--accent:#40c4ff;--ok:#55d68b;--warn:#ffcc66;--danger:#ff6b6b}
    *{box-sizing:border-box} body{margin:0;background:var(--bg);color:#eef3f8;font:15px/1.45 system-ui,"Microsoft YaHei",sans-serif}
    header{position:sticky;top:0;z-index:5;display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:10px 14px;background:#151a22;border-bottom:1px solid var(--line)}
    button,select{border:1px solid var(--line);border-radius:7px;background:#222a35;color:#eef3f8;padding:8px 12px;cursor:pointer}
    button:hover{border-color:var(--accent)} button.primary{background:#087da9;border-color:#18aee7} button.ok{background:#207546;border-color:#43bd79}
    button.danger{background:#723030;border-color:#b74b4b} button:disabled{opacity:.45;cursor:not-allowed}
    #layout{display:grid;grid-template-columns:minmax(0,1fr) 320px;min-height:calc(100vh - 62px)}
    #viewport{overflow:auto;padding:14px;display:flex;align-items:flex-start;justify-content:center;background:#0a0d11}
    #canvas{display:block;box-shadow:0 0 0 1px #455;cursor:crosshair;touch-action:none}
    aside{padding:16px;background:var(--panel);border-left:1px solid var(--line)}
    h2{font-size:17px;margin:0 0 12px}.card{border:1px solid var(--line);border-radius:9px;padding:12px;margin-bottom:12px;background:#171c24}
    .row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.grow{flex:1}.muted{color:#aab5c2}.okText{color:var(--ok)}.warnText{color:var(--warn)}.dangerText{color:var(--danger)}
    #fileName{font-weight:700;word-break:break-all} #boxList{max-height:240px;overflow:auto;padding-left:22px} #boxList li{padding:3px;cursor:pointer} #boxList li.selected{color:var(--accent);font-weight:700}
    kbd{background:#252e3a;border:1px solid #465363;border-bottom-width:2px;border-radius:4px;padding:1px 5px;font-size:12px}
    #toast{position:fixed;left:50%;bottom:20px;transform:translateX(-50%);padding:10px 16px;border-radius:8px;background:#26313e;box-shadow:0 8px 25px #0009;display:none;z-index:10}
    @media(max-width:900px){#layout{grid-template-columns:1fr}aside{border-left:0;border-top:1px solid var(--line)}}
  </style>
</head>
<body>
<header>
  <button id="prevBtn">← 上一张</button><button id="nextBtn">下一张 →</button>
  <select id="filter"><option value="unverified">只看未确认</option><option value="included">全部启用图</option><option value="all">全部图片</option></select>
  <label>缩放 <select id="zoom"><option value="0.75">75%</option><option value="1" selected>100%</option><option value="1.5">150%</option><option value="2">200%</option></select></label>
  <span id="progress" class="muted"></span><span class="grow"></span>
  <button id="saveBtn" class="primary">保存</button><button id="verifyBtn" class="ok">确认并下一张</button>
</header>
<main id="layout">
  <section id="viewport"><canvas id="canvas"></canvas></section>
  <aside>
    <div class="card"><div id="fileName">加载中…</div><div id="stateText" class="muted"></div></div>
    <div class="card">
      <h2>标注框</h2>
      <div class="row"><button id="deleteBtn" class="danger">删除选中框</button><button id="clearBtn">清空框</button></div>
      <ol id="boxList"></ol>
      <div class="muted">在图上按住鼠标拖动即可新增框；单击已有框可选中。</div>
    </div>
    <div class="card">
      <h2>是否用于训练</h2>
      <label><input id="included" type="checkbox"> 启用这张图片</label>
      <div class="muted" style="margin-top:6px">运动模糊、钢球被截断或看不清时取消勾选。原图不会被删除。</div>
    </div>
    <div class="card">
      <h2>快捷键</h2>
      <div><kbd>A</kbd>/<kbd>←</kbd> 上一张　<kbd>D</kbd>/<kbd>→</kbd> 下一张</div>
      <div><kbd>Delete</kbd> 删除框　<kbd>V</kbd> 确认并下一张</div>
      <div><kbd>E</kbd> 启用/排除　<kbd>S</kbd> 保存</div>
    </div>
    <div class="card">
      <h2>导出 YOLO 数据集</h2>
      <p class="muted">仅导出“已确认且启用”的图片。所有框的类别均为 <code>steel_ball</code>。</p>
      <button id="exportBtn">导出 train/val/test</button>
      <div id="exportText" class="muted" style="margin-top:8px"></div>
    </div>
  </aside>
</main>
<div id="toast"></div>
<script>
const canvas=document.querySelector('#canvas'),ctx=canvas.getContext('2d'),viewport=document.querySelector('#viewport');
const state={records:[],visible:[],index:0,current:null,img:new Image(),boxes:[],selected:-1,drag:null,dirty:false};
const $=s=>document.querySelector(s);
function toast(t){const e=$('#toast');e.textContent=t;e.style.display='block';setTimeout(()=>e.style.display='none',1400)}
async function api(url,options){const r=await fetch(url,options);if(!r.ok)throw new Error(await r.text());return r.json()}
function rebuildVisible(keepName){const f=$('#filter').value;state.visible=state.records.filter(r=>f==='all'||(f==='included'&&r.included)||(f==='unverified'&&r.included&&!r.verified));if(!state.visible.length)state.visible=state.records;let i=state.visible.findIndex(r=>r.name===keepName);state.index=i>=0?i:Math.min(state.index,state.visible.length-1);}
async function init(){const data=await api('/api/images');state.records=data.records;rebuildVisible();await loadCurrent()}
async function save(silent=false){if(!state.current||!state.dirty)return;const payload={boxes:state.boxes,included:$('#included').checked,verified:state.current.verified};const saved=await api('/api/annotation/'+encodeURIComponent(state.current.name),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});Object.assign(state.current,saved);const master=state.records.find(r=>r.name===saved.name);Object.assign(master,saved);state.dirty=false;if(!silent)toast('已保存')}
async function loadCurrent(){if(!state.visible.length)return;await save(true);state.current=state.visible[state.index];const full=await api('/api/annotation/'+encodeURIComponent(state.current.name));Object.assign(state.current,full);state.boxes=full.boxes.map(b=>b.slice());state.selected=-1;state.dirty=false;$('#included').checked=full.included;state.img.onload=()=>{canvas.width=state.img.naturalWidth;canvas.height=state.img.naturalHeight;applyZoom();draw()};state.img.src='/image/'+encodeURIComponent(full.name)+'?v='+Date.now();renderSide()}
function applyZoom(){const z=Number($('#zoom').value);canvas.style.width=(canvas.width*z)+'px';canvas.style.height=(canvas.height*z)+'px'}
function renderSide(){const r=state.current;if(!r)return;$('#fileName').textContent=r.name;$('#stateText').innerHTML=(r.verified?'<span class="okText">已确认</span>':'<span class="warnText">未确认</span>')+(r.included?' · 启用':' · <span class="dangerText">已排除</span>')+' · 自动框 '+r.auto_count;$('#progress').textContent=`${state.index+1}/${state.visible.length}　全库 ${state.records.filter(x=>x.verified).length}/${state.records.length} 已确认`;$('#boxList').innerHTML=state.boxes.map((b,i)=>`<li data-i="${i}" class="${i===state.selected?'selected':''}">框 ${i+1}: ${b.map(x=>Math.round(x)).join(', ')}</li>`).join('');document.querySelectorAll('#boxList li').forEach(li=>li.onclick=()=>{state.selected=Number(li.dataset.i);draw();renderSide()})}
function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(state.img,0,0);state.boxes.forEach((b,i)=>{ctx.strokeStyle=i===state.selected?'#40c4ff':'#ff4d4d';ctx.lineWidth=i===state.selected?4:3;ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);ctx.fillStyle=ctx.strokeStyle;ctx.font='bold 18px sans-serif';ctx.fillText(String(i+1),b[0]+3,Math.max(18,b[1]-4))});if(state.drag){const d=state.drag;ctx.strokeStyle='#55d68b';ctx.lineWidth=3;ctx.setLineDash([8,5]);ctx.strokeRect(Math.min(d.x0,d.x1),Math.min(d.y0,d.y1),Math.abs(d.x1-d.x0),Math.abs(d.y1-d.y0));ctx.setLineDash([])}}
function point(ev){const r=canvas.getBoundingClientRect();return{x:(ev.clientX-r.left)*canvas.width/r.width,y:(ev.clientY-r.top)*canvas.height/r.height}}
function hit(x,y){for(let i=state.boxes.length-1;i>=0;i--){const b=state.boxes[i];if(x>=b[0]&&x<=b[2]&&y>=b[1]&&y<=b[3])return i}return-1}
canvas.onpointerdown=e=>{const p=point(e),h=hit(p.x,p.y);if(h>=0){state.selected=h;renderSide();draw();return}state.selected=-1;state.drag={x0:p.x,y0:p.y,x1:p.x,y1:p.y};canvas.setPointerCapture(e.pointerId);draw()};
canvas.onpointermove=e=>{if(!state.drag)return;const p=point(e);state.drag.x1=p.x;state.drag.y1=p.y;draw()};
canvas.onpointerup=e=>{if(!state.drag)return;const d=state.drag;state.drag=null;const b=[Math.max(0,Math.min(d.x0,d.x1)),Math.max(0,Math.min(d.y0,d.y1)),Math.min(canvas.width,Math.max(d.x0,d.x1)),Math.min(canvas.height,Math.max(d.y0,d.y1))];if(b[2]-b[0]>=5&&b[3]-b[1]>=5){state.boxes.push(b);state.selected=state.boxes.length-1;state.dirty=true}draw();renderSide()};
async function move(delta){await save(true);state.index=(state.index+delta+state.visible.length)%state.visible.length;await loadCurrent()}
async function verify(){state.current.verified=true;state.dirty=true;await save(true);const old=state.current.name;rebuildVisible(old);if($('#filter').value==='unverified'){state.index=Math.min(state.index,state.visible.length-1)}else state.index=(state.index+1)%state.visible.length;await loadCurrent();toast('已确认')}
function deleteSelected(){if(state.selected<0)return;state.boxes.splice(state.selected,1);state.selected=-1;state.dirty=true;draw();renderSide()}
$('#prevBtn').onclick=()=>move(-1);$('#nextBtn').onclick=()=>move(1);$('#saveBtn').onclick=()=>save();$('#verifyBtn').onclick=verify;$('#deleteBtn').onclick=deleteSelected;$('#clearBtn').onclick=()=>{if(confirm('清空当前图片的全部框？')){state.boxes=[];state.selected=-1;state.dirty=true;draw();renderSide()}};
$('#included').onchange=()=>{state.current.included=$('#included').checked;state.dirty=true;renderSide()};$('#zoom').onchange=applyZoom;$('#filter').onchange=async()=>{await save(true);rebuildVisible(state.current?.name);await loadCurrent()};
$('#exportBtn').onclick=async()=>{await save(true);$('#exportText').textContent='正在导出…';try{const x=await api('/api/export',{method:'POST'});$('#exportText').textContent=`已导出 ${x.total} 张：train ${x.train} / val ${x.val} / test ${x.test}`;toast('导出完成')}catch(e){$('#exportText').textContent='导出失败：'+e.message}};
window.onkeydown=e=>{if(['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName))return;if(e.key==='ArrowLeft'||e.key.toLowerCase()==='a')move(-1);else if(e.key==='ArrowRight'||e.key.toLowerCase()==='d')move(1);else if(e.key==='Delete')deleteSelected();else if(e.key.toLowerCase()==='v')verify();else if(e.key.toLowerCase()==='s')save();else if(e.key.toLowerCase()==='e'){$('#included').checked=!$('#included').checked;$('#included').onchange()}};
window.onbeforeunload=e=>{if(state.dirty){e.preventDefault();e.returnValue=''}};init().catch(e=>alert(e.message));
</script>
</body></html>"""


DEFAULT_EXCLUDED = {76, 80, 81, 82, 83, 84, 88, 90, 98, 109}
KNOWN_NEGATIVE_RANGE = range(297, 328)


def image_number(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def conservative_boxes(path: Path) -> tuple[int, int, list[list[float]]]:
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"无法读取图片：{path}")
    height, width = image.shape[:2]
    number = image_number(path)
    if number in KNOWN_NEGATIVE_RANGE:
        return width, height, []

    work_width = 640
    scale = work_width / width
    work_height = round(height * scale)
    work = cv2.resize(image, (work_width, work_height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT_ALT,
        dp=1.5,
        minDist=8,
        param1=180,
        param2=0.80,
        minRadius=4,
        maxRadius=34,
    )
    boxes: list[list[float]] = []
    if circles is not None:
        inverse = 1.0 / scale
        for x, y, radius in circles[0]:
            radius = float(radius) * inverse * 1.25
            x, y = float(x) * inverse, float(y) * inverse
            boxes.append([
                max(0.0, x - radius),
                max(0.0, y - radius),
                min(float(width), x + radius),
                min(float(height), y + radius),
            ])
    return width, height, boxes


def initialize(source: Path, state_path: Path) -> dict:
    records = {}
    for path in sorted(source.glob("steel_ball_*.jpg")):
        number = image_number(path)
        width, height, boxes = conservative_boxes(path)
        excluded = number in DEFAULT_EXCLUDED
        known_negative = number in KNOWN_NEGATIVE_RANGE
        records[path.name] = {
            "name": path.name,
            "width": width,
            "height": height,
            "boxes": boxes,
            "auto_count": len(boxes),
            "included": not excluded,
            "verified": bool(excluded or known_negative),
        }
    state = {"version": 1, "records": records}
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state


def atomic_write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def split_name(name: str) -> str:
    number = int(Path(name).stem.rsplit("_", 1)[1])
    # Negatives were photographed in short, highly similar sequences. Keep each
    # sequence together while ensuring every split contains hard negatives.
    if 297 <= number <= 315:
        return ("train", "val", "test", "train")[(number - 297) // 5]
    if 316 <= number <= 327:
        return ("train", "val", "test")[(number - 316) // 4]
    # Mixed AirPods-case + steel-ball frames form one hard-confuser test scene.
    if 328 <= number <= 332:
        return "test"
    group = number // 5
    value = int(hashlib.sha1(f"steel-ball-2026:{group}".encode()).hexdigest()[:8], 16) % 100
    if value < 75:
        return "train"
    if value < 90:
        return "val"
    return "test"


def export_yolo(source: Path, dataset_root: Path, state: dict) -> dict[str, int]:
    selected = [r for r in state["records"].values() if r.get("included") and r.get("verified")]
    counts = {"train": 0, "val": 0, "test": 0}
    for record in selected:
        split = split_name(record["name"])
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        source_path = source / record["name"]
        shutil.copy2(source_path, image_dir / record["name"])
        width, height = float(record["width"]), float(record["height"])
        lines = []
        for x1, y1, x2, y2 in record["boxes"]:
            x1, x2 = sorted((max(0.0, x1), min(width, x2)))
            y1, y2 = sorted((max(0.0, y1), min(height, y2)))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            xc = (x1 + x2) / (2 * width)
            yc = (y1 + y2) / (2 * height)
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height
            lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        (label_dir / (Path(record["name"]).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        counts[split] += 1
    counts["total"] = sum(counts.values())
    return counts


class AnnotatorServer(ThreadingHTTPServer):
    def __init__(self, address, handler, source: Path, dataset_root: Path, state_path: Path, state: dict):
        super().__init__(address, handler)
        self.source = source
        self.dataset_root = dataset_root
        self.state_path = state_path
        self.state = state
        self.lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server: AnnotatorServer

    def log_message(self, fmt: str, *args) -> None:
        return

    def send_bytes(self, data: bytes, content_type: str, status=HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, value: dict, status=HTTPStatus.OK) -> None:
        self.send_bytes(json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_bytes(APP_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/images":
            records = []
            for record in self.server.state["records"].values():
                records.append({key: record[key] for key in ("name", "included", "verified", "auto_count")})
            self.send_json({"records": records})
            return
        if parsed.path.startswith("/api/annotation/"):
            name = unquote(parsed.path.split("/api/annotation/", 1)[1])
            record = self.server.state["records"].get(name)
            if record is None:
                self.send_json({"error": "未找到图片"}, HTTPStatus.NOT_FOUND)
            else:
                self.send_json(record)
            return
        if parsed.path.startswith("/image/"):
            name = Path(unquote(parsed.path.split("/image/", 1)[1])).name
            path = self.server.source / name
            if not path.is_file():
                self.send_json({"error": "未找到图片"}, HTTPStatus.NOT_FOUND)
                return
            self.send_bytes(path.read_bytes(), mimetypes.guess_type(path.name)[0] or "image/jpeg")
            return
        self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/export":
            try:
                counts = export_yolo(self.server.source, self.server.dataset_root, self.server.state)
                self.send_json(counts)
            except Exception as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if parsed.path.startswith("/api/annotation/"):
            name = unquote(parsed.path.split("/api/annotation/", 1)[1])
            record = self.server.state["records"].get(name)
            if record is None:
                self.send_json({"error": "未找到图片"}, HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                boxes = payload.get("boxes", [])
                clean_boxes = []
                for box in boxes:
                    if len(box) != 4:
                        continue
                    clean_boxes.append([float(value) for value in box])
                with self.server.lock:
                    record["boxes"] = clean_boxes
                    record["included"] = bool(payload.get("included", True))
                    record["verified"] = bool(payload.get("verified", False))
                    atomic_write_json(self.server.state_path, self.server.state)
                self.send_json(record)
            except Exception as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)


def main() -> None:
    parser = argparse.ArgumentParser(description="小铁球 YOLO 标注复核工具")
    parser.add_argument("--source", type=Path, required=True, help="原始 JPG 图片目录")
    parser.add_argument("--dataset", type=Path, required=True, help="YOLO 数据集根目录")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--reinitialize", action="store_true", help="重新生成预标注，会覆盖现有复核进度")
    parser.add_argument("--export-only", action="store_true", help="按当前复核状态导出后退出")
    args = parser.parse_args()
    source = args.source.resolve()
    dataset_root = args.dataset.resolve()
    state_path = dataset_root / "annotations" / "steel_ball_annotations.json"
    if not source.is_dir():
        raise SystemExit(f"图片目录不存在：{source}")
    if args.reinitialize or not state_path.exists():
        print("正在生成保守预标注，请稍候……")
        state = initialize(source, state_path)
    else:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    print(f"已载入 {len(state['records'])} 张图片")
    if args.export_only:
        counts = export_yolo(source, dataset_root, state)
        print(json.dumps(counts, ensure_ascii=False))
        return
    print(f"浏览器地址：http://127.0.0.1:{args.port}")
    server = AnnotatorServer(("127.0.0.1", args.port), Handler, source, dataset_root, state_path, state)
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(f"http://127.0.0.1:{args.port}")).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止标注工具。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
