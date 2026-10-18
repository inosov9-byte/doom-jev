"""Build a local HTML viewer for a comparison folder (runs/compare-*).

Views: per-arm results, one run with its video synced to the decision timeline (command,
latency, probabilities, the exact state text the engine read), and two runs side by side.
Videos are linked by relative path, so open index.html from the folder itself.

usage: python -m jevdoom.report runs/compare-20261018-010000-live
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ARM_LABEL = {"jev-local": "Jev local (Qwen 0.8B, laptop GPU)", "jev-hosted": "Jev hosted (Qwen3.6-35B)",
             "jev-hosted-gemma": "Jev hosted (Gemma-4-26B)", "haiku": "Claude Haiku 5.5",
             "haiku-think": "Claude Haiku 5.5 + thinking", "rules": "Rules expert (teacher, instant)",
             "rules-lag": "Rules expert at 0.7 s per decision"}


def load_runs(folder: Path) -> list[dict]:
    runs = []
    for s in sorted(folder.glob("*/summary.json")):
        summ = json.loads(s.read_text())
        decisions = []
        for line in (s.parent / "decisions.jsonl").open(encoding="utf-8"):
            d = json.loads(line)
            probs = d.get("probabilities") or {}
            top = sorted(probs.items(), key=lambda kv: -kv[1])[:5]
            decisions.append({"tic": d["tic"], "obs_tic": d["obs_tic"], "choice": d.get("choice"),
                              "latency": d.get("latency"), "error": d.get("error"), "health": d.get("health"),
                              "state": d["state"], "top": [[k, round(v, 3)] for k, v in top]})
        video = s.parent / "run.mp4"
        runs.append({"id": s.parent.name, "summary": summ, "decisions": decisions,
                     "video": f"{s.parent.name}/run.mp4" if video.exists() else None})
    return runs


def build(folder: Path, title: str = "Jev vs Haiku: real-time DOOM E1M1") -> Path:
    runs = load_runs(folder)
    res = folder / "results.json"
    meta = json.loads(res.read_text()).get("meta", {}) if res.exists() else {}
    agg = json.loads(res.read_text()).get("aggregate", []) if res.exists() else []
    probe = []
    for p in sorted(folder.parent.glob("compare-*-probe/results.json")):
        pr = json.loads(p.read_text())
        if not pr.get("meta", {}).get("dry_run"):
            probe = pr.get("rows", probe)
    data = {"meta": meta, "aggregate": agg, "probe": probe, "runs": runs, "labels": ARM_LABEL}
    html = TEMPLATE.replace("/*DATA*/null", json.dumps(data)).replace("__TITLE__", title)
    out = folder / "index.html"
    out.write_text(html, encoding="utf-8")
    return out


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jev vs Haiku</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--line:#2a2f3a;--text:#e6e8ec;--muted:#9aa3b2;--accent:#6aa8ff;
--move:#4cc38a;--turn:#6aa8ff;--fire:#ff6b6b;--use:#f5c451;--err:#888}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,Segoe UI,sans-serif}
header{padding:18px 24px;border-bottom:1px solid var(--line);display:flex;gap:24px;align-items:baseline;flex-wrap:wrap}
h1{font-size:20px;margin:0}nav button{background:none;border:1px solid var(--line);color:var(--text);padding:6px 12px;border-radius:6px;cursor:pointer;margin-right:6px}
nav button.on{border-color:var(--accent);color:var(--accent)}main{padding:20px 24px;max-width:1500px}
table{border-collapse:collapse;width:100%;margin:8px 0 24px}th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:500}tr.click{cursor:pointer}tr.click:hover{background:var(--panel)}
.bar{display:inline-block;height:8px;background:var(--accent);border-radius:2px;vertical-align:middle;margin-left:6px}
.muted{color:var(--muted)}.grid{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(320px,1fr);gap:18px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
video{width:100%;border-radius:8px;background:#000}canvas{width:100%;height:90px;display:block;cursor:pointer}
.cmd{font:600 26px/1.2 ui-monospace,Consolas,monospace}.state{font:12px/1.5 ui-monospace,Consolas,monospace;color:#c9cfda;white-space:pre-wrap}
.prob{display:flex;align-items:center;gap:8px;font:12px ui-monospace,Consolas,monospace}.prob span:first-child{width:130px}
.prob i{display:inline-block;height:9px;background:var(--accent);border-radius:2px}.legend span{margin-right:14px}
.dot{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
select{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:5px}
.sbs{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:900px){.grid,.sbs{grid-template-columns:1fr}}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;border:1px solid var(--line)}
.exit{color:var(--move);border-color:var(--move)}.died{color:var(--fire);border-color:var(--fire)}.timeout{color:var(--use);border-color:var(--use)}
</style></head><body>
<header><h1>__TITLE__</h1><span class="muted" id="meta"></span>
<nav><button data-v="results" class="on">Results</button><button data-v="run">Run</button><button data-v="sbs">Side by side</button></nav></header>
<main>
<section id="v-results"></section>
<section id="v-run" hidden>
 <div style="margin-bottom:10px"><select id="runsel"></select></div>
 <div class="grid"><div class="card"><video id="vid" controls muted></video>
  <canvas id="tl" width="1200" height="90"></canvas>
  <div class="legend muted" style="margin-top:6px"><span><i class="dot" style="background:var(--move)"></i>move</span><span><i class="dot" style="background:var(--turn)"></i>turn</span><span><i class="dot" style="background:var(--fire)"></i>fire</span><span><i class="dot" style="background:var(--use)"></i>use</span><span>bar height = decision latency (top = 3 s)</span></div>
  <div id="runstats" class="muted" style="margin-top:8px"></div></div>
  <div class="card"><div class="muted" id="dhead"></div><div class="cmd" id="dcmd">-</div><div id="dprob" style="margin:10px 0"></div>
  <div class="muted" style="margin-bottom:4px">What the engine read</div><div class="state" id="dstate"></div></div></div>
</section>
<section id="v-sbs" hidden>
 <div style="margin-bottom:10px">Seed <select id="seedsel"></select> &nbsp; Left <select id="lsel"></select> &nbsp; Right <select id="rsel"></select> &nbsp; <button id="play">Play both</button></div>
 <div class="sbs"><div class="card"><div id="lhead"></div><video id="lvid" muted></video><div class="cmd" id="lcmd"></div><div class="muted" id="llat"></div></div>
 <div class="card"><div id="rhead"></div><video id="rvid" muted></video><div class="cmd" id="rcmd"></div><div class="muted" id="rlat"></div></div></div>
</section>
</main>
<script>
const D=/*DATA*/null;const L=a=>D.labels[a]||a;const $=id=>document.getElementById(id);
const kind=c=>!c?"err":c==="fire"?"fire":c==="use"?"use":c.startsWith("turn")?"turn":"move";
const col={move:"#4cc38a",turn:"#6aa8ff",fire:"#ff6b6b",use:"#f5c451",err:"#888"};
$("meta").textContent=`${D.runs.length} runs · seeds ${(D.meta.seeds||[]).join(", ")} · commit ${D.meta.commit||"?"}${D.meta.dry_run?" · DRY RUN (mock engines)":""}`;
document.querySelectorAll("nav button").forEach(b=>b.onclick=()=>{document.querySelectorAll("nav button").forEach(x=>x.classList.toggle("on",x===b));
 ["results","run","sbs"].forEach(v=>$("v-"+v).hidden=v!==b.dataset.v)});
function results(){let h="<h2>Per arm (medians over seeds)</h2><table><tr><th>Arm</th><th>Runs</th><th>Exit rate</th><th>Route progress</th><th>Survival s</th><th>Kills</th><th>Latency median</th><th>p90</th><th>Decisions/s</th><th>Errors</th><th>Cost $</th></tr>";
 for(const a of D.aggregate){h+=`<tr><td>${L(a.arm)}</td><td>${a.runs}</td><td>${Math.round(a.exit_rate*100)}%</td><td>${a.progress_median_pct}%<span class="bar" style="width:${a.progress_median_pct}px"></span></td><td>${a.survival_median_s}</td><td>${a.kills_median}</td><td>${a.latency_median_s}s<span class="bar" style="width:${a.latency_median_s*40}px;background:var(--use)"></span></td><td>${a.latency_p90_s}s</td><td>${a.decisions_per_s}</td><td>${a.errors}</td><td>${a.cost_usd_total}</td></tr>`}
 h+="</table>";if(D.probe.length){const ks=Object.keys(D.probe[D.probe.length-1].by_kind||{});h+="<h2>Decision quality without time pressure (60-situation probe)</h2><table><tr><th>Arm</th><th>Score</th><th>Wrong-way turns</th>"+ks.map(k=>`<th>${k}</th>`).join("")+"</tr>";
 for(const p of D.probe){h+=`<tr><td>${L(p.arm)}</td><td>${p.score}</td><td>${p.wrong_direction_turns}</td>`+ks.map(k=>`<td>${(p.by_kind||{})[k]??""}</td>`).join("")+"</tr>"}h+="</table>"}
 h+="<h2>Every run</h2><p class='muted'>Click a run to open it.</p><table><tr><th>Run</th><th>Seed</th><th>Outcome</th><th>Progress</th><th>Time s</th><th>Kills</th><th>Latency</th><th>Decisions/s</th><th>Cost $</th></tr>";
 D.runs.forEach((r,i)=>{const s=r.summary;h+=`<tr class="click" data-i="${i}"><td>${L(s.arm)}</td><td>${s.seed}</td><td><span class="tag ${s.outcome}">${s.outcome}</span></td><td>${s.route_progress_pct}%</td><td>${s.game_seconds}</td><td>${s.kills}</td><td>${s.latency_median}s</td><td>${s.decisions_per_s}</td><td>${s.cost_usd}</td></tr>`});
 $("v-results").innerHTML=h+"</table>";document.querySelectorAll("tr.click").forEach(t=>t.onclick=()=>{openRun(+t.dataset.i);document.querySelector('nav button[data-v="run"]').click()})}
let cur=null;
function openRun(i){cur=D.runs[i];$("runsel").value=i;const s=cur.summary;$("vid").src=cur.video||"";
 $("runstats").innerHTML=`${L(s.arm)} · seed ${s.seed} · <span class="tag ${s.outcome}">${s.outcome}</span> · ${s.route_progress_pct}% of route · ${s.game_seconds}s · ${s.decisions} decisions (${s.decisions_per_s}/s) · median latency ${s.latency_median}s · kills ${s.kills} · cost $${s.cost_usd}`;
 drawTL();showDecision(0)}
function drawTL(){const c=$("tl"),g=c.getContext("2d"),T=cur.summary.game_seconds*35||1;g.clearRect(0,0,c.width,c.height);g.fillStyle="#1d2129";g.fillRect(0,0,c.width,c.height);
 for(const d of cur.decisions){const x=d.tic/T*c.width,h=Math.min(1,(d.latency||3)/3)*(c.height-10);g.fillStyle=col[kind(d.choice)];g.fillRect(x,c.height-h,Math.max(2,c.width/cur.decisions.length*0.6),h)}
 const t=($("vid").currentTime||0)*35/T*c.width;g.fillStyle="#fff";g.fillRect(t,0,2,c.height)}
function idxAt(tic,ds){let k=0;for(let j=0;j<ds.length;j++){if(ds[j].tic<=tic)k=j;else break}return k}
function showDecision(k){const d=cur.decisions[k];if(!d)return;$("dhead").textContent=`decision ${k+1}/${cur.decisions.length} · t=${(d.tic/35).toFixed(1)}s · latency ${d.latency??"-"}s · health ${d.health}`;
 $("dcmd").textContent=d.choice||("error: "+d.error);$("dcmd").style.color=col[kind(d.choice)];
 $("dprob").innerHTML=d.top.length?d.top.map(([k2,p])=>`<div class="prob"><span>${k2}</span><i style="width:${Math.round(p*200)}px"></i><span>${p.toFixed(2)}</span></div>`).join(""):"<span class='muted'>No probabilities: a generative engine returns one answer.</span>";
 $("dstate").textContent=d.state}
const sync=()=>{if(!cur)return;showDecision(idxAt($("vid").currentTime*35,cur.decisions));drawTL()};
$("vid").addEventListener("timeupdate",sync);$("vid").addEventListener("seeked",sync);
$("tl").onclick=e=>{const r=e.target.getBoundingClientRect(),T=cur.summary.game_seconds;$("vid").currentTime=(e.clientX-r.left)/r.width*T};
D.runs.forEach((r,i)=>{const o=document.createElement("option");o.value=i;o.textContent=`${L(r.summary.arm)} · seed ${r.summary.seed} · ${r.summary.outcome} ${r.summary.route_progress_pct}%`;$("runsel").appendChild(o)});
$("runsel").onchange=e=>openRun(+e.target.value);
const seeds=[...new Set(D.runs.map(r=>r.summary.seed))];seeds.forEach(s=>{const o=document.createElement("option");o.value=s;o.textContent=s;$("seedsel").appendChild(o)});
const arms=[...new Set(D.runs.map(r=>r.summary.arm))];for(const id of["lsel","rsel"])arms.forEach(a=>{const o=document.createElement("option");o.value=a;o.textContent=L(a);$(id).appendChild(o)});
$("lsel").value=arms.includes("jev-local")?"jev-local":arms[0];$("rsel").value=arms.includes("haiku")?"haiku":arms[arms.length-1];
let side={};function loadSide(){for(const [p,sel] of [["l","lsel"],["r","rsel"]]){const r=D.runs.find(x=>x.summary.arm===$(sel).value&&String(x.summary.seed)===$("seedsel").value);side[p]=r;
 $(p+"vid").src=r&&r.video||"";$(p+"head").innerHTML=r?`<b>${L(r.summary.arm)}</b> <span class="tag ${r.summary.outcome}">${r.summary.outcome}</span> ${r.summary.route_progress_pct}% · ${r.summary.game_seconds}s · median latency ${r.summary.latency_median}s`:"<span class='muted'>no run for this seed</span>"}}
for(const p of["l","r"])$(p+"vid").ontimeupdate=$(p+"vid").onseeked=()=>{const r=side[p];if(!r)return;const d=r.decisions[idxAt($(p+"vid").currentTime*35,r.decisions)];if(d){$(p+"cmd").textContent=d.choice||"error";$(p+"cmd").style.color=col[kind(d.choice)];$(p+"lat").textContent=`decision latency ${d.latency??"-"}s`}};
["seedsel","lsel","rsel"].forEach(id=>$(id).onchange=loadSide);$("play").onclick=()=>{for(const p of["l","r"]){$(p+"vid").currentTime=0;$(p+"vid").play()}};
results();if(D.runs.length)openRun(0);loadSide();
</script></body></html>
"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--title", default="Jev vs Haiku: real-time DOOM E1M1")
    args = ap.parse_args(argv)
    print(build(Path(args.folder), args.title))


if __name__ == "__main__":
    main()
