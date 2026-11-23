"""실험 궤적을 레이어(시간 단계)별 노드 그래프로 보는 viewer.html 생성 후 새 창으로 연다.

실행: python3 build_viewer.py [--no-open]
선택 기능: 그래프/집계/동역학 고르기, 5개 레이어씩 페이지 넘기기.
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path

import numpy as np

import rank_experiment as rx

ROOT = Path(__file__).resolve().parent
LAYER_EVERY = 25     # 25단계마다 한 레이어 (t=0,25,...,500 → 21개)
PER_PAGE = 5


def layout(g: rx.G, truth, seed=0, iters=300):
    rng = np.random.default_rng(seed)
    n = g.n
    pos = rng.normal(0, 0.3, (n, 2))
    if truth is not None:
        pos[:, 0] += np.where(truth == 0, -1.0, 1.0)
    A = (g.A > 0).astype(float)
    k = 1.0 / np.sqrt(n) * 2.2
    temp = 0.3
    for _ in range(iters):
        d = pos[:, None] - pos[None]
        dist = np.linalg.norm(d, axis=-1) + 1e-9
        rep = (k * k / dist ** 2)[..., None] * d
        att = -(A * dist / k)[..., None] * d / dist[..., None]
        f = rep.sum(1) + att.sum(1)
        norm = np.linalg.norm(f, axis=1, keepdims=True) + 1e-9
        pos += f / norm * np.minimum(norm, temp)
        temp *= 0.985
    pos -= pos.mean(0)
    pos /= np.abs(pos).max(0) + 1e-9
    return pos


def experiments():
    sp = []
    for kind in ("cycle", "clique", "sbm"):
        for b in (0, 1, 2, 4, 8):
            sp.append((f"{kind} 24+24, 연결 {b}개", dict(kind=kind, sizes=(24, 24), bridges=b, wb=1.0, seed=100, role="v")))
    sp.append(("음성: 무작위 ER", dict(kind="er", sizes=(24, 24), bridges=0, wb=1.0, seed=100, role="v")))
    sp.append(("음성: 재배선 SBM", dict(kind="shuffled", sizes=(24, 24), bridges=0, wb=1.0, seed=100, role="v")))
    return sp


def build():
    params = ast.literal_eval((ROOT / "results" / "params.txt").read_text())
    data = {"graphs": [], "layers": list(range(0, rx.T + 1, LAYER_EVERY))}
    idx = data["layers"]
    for name, spec in experiments():
        g, truth = rx.prepare(spec)
        rng = np.random.default_rng(spec["seed"] * 1000 + 17)
        u0 = rng.uniform(-1, 1, (rx.R, g.n))
        entry = {
            "name": name, "n": g.n,
            "pos": np.round(layout(g, truth), 3).tolist(),
            "edges": [[int(i), int(j), int(truth is not None and truth[i] != truth[j])]
                      for i, j in zip(g.ei, g.ej)],
            "truth": None if truth is None else truth.tolist(),
            "runs": {},
        }
        for agg in rx.AGGS:
            for dyn in rx.DYNS:
                X, failed = rx.simulate(g, agg, dyn, u0, zero_failed=False)  # 폭주 값도 그대로 보존
                variants = [("diff", X)] if dyn == "diff" else \
                    [("wave_raw", X), ("wave_lp", rx.lowpass(X, params["fc"]))]
                for vname, Xv in variants:
                    x = Xv[idx, 0, :]
                    bad = ~(np.abs(Xv[:, 0, :]) < rx.FAIL_AMP).all(axis=1)
                    fstep = int(np.argmax(bad)) if bad.any() else None
                    # 비유한 값(inf/nan)은 null, 매우 큰 값은 float 한계 내에서 그대로 둔다
                    vals = [[round(float(v), 2) if np.isfinite(v) else None for v in row] for row in x]
                    entry["runs"][f"{agg}|{vname}"] = {"failed": bool(failed[0]), "fstep": fstep, "vals": vals}
        data["graphs"].append(entry)
        print("built", name, flush=True)
    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":"))) \
                   .replace("__PER__", str(PER_PAGE))
    out = ROOT / "viewer.html"
    out.write_text(html, encoding="utf-8")
    return out


TEMPLATE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>레이어별 노드 그래프 뷰어</title>
<style>
:root{--bg:#f6f7f9;--fg:#1c2330;--card:#fff;--line:#d9dde4;--mut:#667085;--acc:#2f5fd0}
@media (prefers-color-scheme:dark){:root{--bg:#14171c;--fg:#e6e9ef;--card:#1e232b;--line:#343b47;--mut:#98a2b3;--acc:#7aa2ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,"Apple SD Gothic Neo",sans-serif}
header{padding:14px 16px;border-bottom:1px solid var(--line);background:var(--card);position:sticky;top:0;z-index:2}
h1{font-size:16px;margin:0 0 10px}
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:end}
label{display:flex;flex-direction:column;font-size:12px;color:var(--mut);gap:3px}
select,button{font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg)}
button{cursor:pointer}button:disabled{opacity:.4;cursor:default}button.p{background:var(--acc);color:#fff;border-color:var(--acc)}
.pager{display:flex;gap:8px;align-items:center;margin-left:auto}
#info{padding:8px 16px;color:var(--mut);font-size:13px}
.warn{color:#c2410c;font-weight:600}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px;padding:0 16px 24px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px}
.stat{font-size:12px;color:var(--mut);margin-bottom:4px}
.card h3{font-size:13px;margin:0 0 4px}
svg{width:100%;height:auto;display:block}
.legend{display:flex;align-items:center;gap:6px;padding:0 16px 8px;font-size:12px;color:var(--mut)}
.bar{width:160px;height:10px;border-radius:5px;background:linear-gradient(90deg,#2563eb,#f8fafc,#dc2626)}
</style></head><body>
<header><h1>레이어(시간 단계)별 노드 그래프</h1>
<div class="controls">
<label>그래프<select id="g"></select></label>
<label>집계<select id="a"><option value="mean">Mean</option><option value="trim">Trim</option><option value="owa">OWA</option><option value="vw">Value-weight</option></select></label>
<label>이상치 기준(표준편차)<select id="z"><option value="1.5">1.5σ</option><option value="2" selected>2σ</option><option value="3">3σ</option></select></label>
<label>동역학<select id="d"><option value="diff">확산</option><option value="wave_raw">파동 (원시)</option><option value="wave_lp">파동 (저주파)</option></select></label>
<div class="pager"><button id="first">⏮</button><button id="prev" class="p">◀ 이전</button><span id="pg"></span><button id="next" class="p">다음 ▶</button><button id="last">⏭</button></div>
</div></header>
<div id="info"></div>
<div class="legend">노드 값 <span>음수</span><span class="bar"></span><span>양수</span><span id="rng"></span> · 노란 굵은 선 = 연결 간선 · 빨간 점선 원 = 레이어 평균에서 기준σ 이상 벗어난 노드</div>
<div id="grid"></div>
<script>
const D=__DATA__, PER=__PER__;
const $=id=>document.getElementById(id);
let page=0;let Z=2;
D.graphs.forEach((g,i)=>$("g").add(new Option(g.name,i)));
function color(v,m){if(v===null)return"#111";const sl=x=>Math.sign(x)*Math.log1p(Math.abs(x));const t=Math.max(-1,Math.min(1,sl(v)/sl(m)));const a=Math.abs(t);
 const c=t<0?[37,99,235]:[220,38,38],w=[248,250,252];
 return `rgb(${w.map((x,k)=>Math.round(x+(c[k]-x)*a)).join(",")})`}
function render(){
 const g=D.graphs[+$("g").value], key=$("a").value+"|"+$("d").value, run=g.runs[key];
 if(!run){$("grid").innerHTML="";$("info").textContent="이 조합은 데이터가 없습니다.";return}
 const L=D.layers, pages=Math.ceil(L.length/PER);
 page=Math.max(0,Math.min(page,pages-1));
 let m=0;run.vals.forEach(r=>r.forEach(v=>{if(v!==null)m=Math.max(m,Math.abs(v))}));m=m||1;
 $("rng").textContent=`(부호 로그 스케일, 최대 ±${m>1e4?m.toExponential(1):m.toFixed(2)})`;
 $("info").innerHTML=`노드 ${g.n}개 · 간선 ${g.edges.length}개 · 레이어 ${L.length}개 (${L[0]}~${L[L.length-1]}단계, ${L[1]-L[0]}단계 간격)`+
  (run.failed?` · <span class="warn">⚠ 값 폭주(|u|>20) ${run.fstep!==null?'t='+run.fstep+'부터':''} — 값은 그대로 표시, 검은 노드는 ∞/NaN</span>`:"");
 $("pg").textContent=`${page+1} / ${pages}`;
 $("first").disabled=$("prev").disabled=page===0;$("last").disabled=$("next").disabled=page===pages-1;
 const S=240,pad=14,P=g.pos.map(p=>[pad+(p[0]+1)/2*(S-2*pad),pad+(p[1]+1)/2*(S-2*pad)]);
 const edges=g.edges.map(e=>`<line x1="${P[e[0]][0]}" y1="${P[e[0]][1]}" x2="${P[e[1]][0]}" y2="${P[e[1]][1]}" stroke="${e[2]?'#f59e0b':'#9aa3b2'}" stroke-width="${e[2]?2.4:.7}" opacity="${e[2]?1:.55}"/>`).join("");
 let h="";
 for(let li=page*PER;li<Math.min(L.length,(page+1)*PER);li++){
  const row=run.vals[li],fin=row.filter(v=>v!==null),mu=fin.reduce((a,b)=>a+b,0)/(fin.length||1);
  const sd=Math.sqrt(fin.reduce((a,b)=>a+(b-mu)**2,0)/(fin.length||1));
  const out=row.map(v=>v===null||(sd>1e-9&&Math.abs(v-mu)>Z*sd));
  const nodes=P.map((p,i)=>(out[i]?`<circle cx="${p[0]}" cy="${p[1]}" r="10" fill="none" stroke="#e11d48" stroke-width="2" stroke-dasharray="3 2"/>`:"")+
   `<circle cx="${p[0]}" cy="${p[1]}" r="5.5" fill="${color(row[i],m)}" stroke="${g.truth&&g.truth[i]?'#0f766e':'#7c3aed'}" stroke-width="1.2"/>`).join("");
  const no=out.filter(Boolean).length,fm=v=>Math.abs(v)>=1e4?v.toExponential(1):v.toFixed(2);
  h+=`<div class="card"><h3>레이어 ${li+1} · t = ${L[li]}</h3><div class="stat">평균 ${fm(mu)} · σ ${fm(sd)} · <b class="${no?'warn':''}">이상치 ${no}개</b></div><svg viewBox="0 0 ${S} ${S}">${edges}${nodes}</svg></div>`;
 }
 $("grid").innerHTML=h;
}
["g","a","d"].forEach(id=>$(id).addEventListener("change",()=>{page=0;render()}));
$("z").addEventListener("change",()=>{Z=+$("z").value;render()});
$("prev").onclick=()=>{page--;render()};$("next").onclick=()=>{page++;render()};
$("first").onclick=()=>{page=0;render()};$("last").onclick=()=>{page=1e9;render()};
document.addEventListener("keydown",e=>{if(e.key==="ArrowLeft"){page--;render()}if(e.key==="ArrowRight"){page++;render()}});
render();
</script></body></html>
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    out = build()
    print("wrote", out, f"({out.stat().st_size/1e6:.2f} MB)")
    if not a.no_open:
        subprocess.run(["open", str(out)], check=False)   # macOS: 기본 브라우저의 새 창/탭
