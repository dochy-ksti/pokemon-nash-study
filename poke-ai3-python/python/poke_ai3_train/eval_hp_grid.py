"""HP 4 軸層別 (active 5×5 × bench 5×5) の交代確率をドリルダウン HTML で出力する。

eval-hp-strategy が出す active HP の 5×5 を「外側」とし、各セルをクリックすると
その局面をさらに自分 bench HP × 相手 bench HP で層別した「内側」5×5 が開く、
自己完結 HTML ビューアを生成する。データは self-play を回して収集する。

起動例:
  uv run eval-hp-grid --checkpoint data/poke-ai3/stage3c_weak_s1.pt --stage 3c \\
      --num-eval-games 4000 --out data/poke-ai3/hp_grid_3c.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import Agent
from .diagnostics import HP_BUCKET_LABELS, SWITCH_MATCHUP_KEYS, hp_4d_switch_diagnostics
from .encoding import encode_observations
from .eval_hp_strategy import collect_examples
from .train_loop import get_rust_async_executor_wrapper

_HTML_TEMPLATE = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<title>HP 交代率グリッド __TITLE__</title>
<style>
 body{font-family:sans-serif;margin:16px;background:#fafafa}
 h1{font-size:18px} .meta{color:#555;font-size:13px;margin-bottom:12px}
 .matchups{margin-bottom:12px} button.mk{margin-right:6px;padding:4px 10px;cursor:pointer}
 button.mk.active{background:#1565c0;color:#fff}
 .panes{display:flex;gap:32px;flex-wrap:wrap}
 table{border-collapse:collapse;margin-top:6px} caption{font-weight:bold;padding:4px}
 td,th{border:1px solid #bbb;width:64px;height:40px;text-align:center;font-size:13px}
 th{background:#eee}
 td.cell{cursor:pointer} td.cell.sel{outline:3px solid #d32f2f}
 td.na{color:#aaa;background:#f3f3f3;cursor:default}
 .legend{font-size:12px;color:#555;margin-top:8px}
 .agg{font-size:12px;color:#333;margin-top:4px}
</style></head><body>
<h1>HP 交代率グリッド __TITLE__</h1>
<div class="meta" id="meta"></div>
<div class="matchups" id="matchups"></div>
<div class="panes">
 <div><div id="outerWrap"></div><div class="agg" id="outerAgg"></div></div>
 <div><div id="innerWrap"></div><div class="agg" id="innerAgg"></div></div>
</div>
<div class="legend">外側=active HP (行:自分 / 列:相手)。セルをクリックすると、その局面を
 bench HP で層別した内側 5×5 (行:自分 bench / 列:相手 bench) を表示。色=交代確率
 (青=居座り 0 〜 赤=交代 1)。n&lt;min_n のセルは "--"。値は model の交代確率。</div>
<script>
const DATA = __DATA__;
const L = DATA.labels, MINN = DATA.min_n;
// cells: [matchup,a,b,c,d,model,teacher,n]
let curMk = DATA.matchups[0], curOuter = null;
function color(p){ // 0->青, 1->赤
 const r=Math.round(255*p), b=Math.round(255*(1-p));
 return `rgb(${r},${Math.round(60+60*(1-Math.abs(p-0.5)*2))},${b})`;
}
function aggOuter(mk){ // {a,b: {m,t,n}}
 const o={};
 for(const c of DATA.cells){ if(c[0]!==mk) continue;
  const k=c[1]+","+c[2]; const e=o[k]||(o[k]={m:0,t:0,n:0});
  e.m+=c[5]*c[7]; e.t+=c[6]*c[7]; e.n+=c[7]; }
 return o;
}
function innerCells(mk,a,b){ // {c,d:{m,t,n}}
 const o={};
 for(const c of DATA.cells){ if(c[0]!==mk||c[1]!==a||c[2]!==b) continue;
  o[c[3]+","+c[4]]={m:c[5],t:c[6],n:c[7]}; }
 return o;
}
function grid(title,getCell,selKey){
 let h=`<table><caption>${title}</caption><tr><th>自\\相</th>`;
 for(const l of L) h+=`<th>${l}</th>`; h+="</tr>";
 for(let i=0;i<5;i++){ h+=`<tr><th>${L[i]}</th>`;
  for(let j=0;j<5;j++){ const e=getCell(i,j); const key=i+","+j;
   if(!e||e.n<MINN){ const nn=e?` (${e.n})`:""; h+=`<td class="na">--${nn}</td>`; }
   else{ const sel=(selKey===key)?" sel":"";
    h+=`<td class="cell${sel}" style="background:${color(e.show)}" data-k="${key}">${e.show.toFixed(2)}<br><span style="font-size:10px;color:#fff">n=${e.n}</span></td>`; }
  } h+="</tr>"; }
 h+="</table>";
 return h;
}
function renderOuter(){
 const o=aggOuter(curMk);
 const get=(i,j)=>{ const e=o[i+","+j]; if(!e)return null;
  return {n:e.n, show:e.m/e.n}; };
 document.getElementById("outerWrap").innerHTML=
  grid("外側: active HP — "+curMk, get, curOuter);
 let tot=0,ms=0,ts=0; for(const k in o){const e=o[k];tot+=e.n;ms+=e.m;ts+=e.t;}
 document.getElementById("outerAgg").innerHTML= tot?
  `集計 n=${tot} / model=${(ms/tot).toFixed(3)} / teacher=${(ts/tot).toFixed(3)}`:"データなし";
 document.querySelectorAll("#outerWrap td.cell").forEach(td=>{
  td.onclick=()=>{ curOuter=td.dataset.k; renderOuter(); renderInner(); };
 });
}
function renderInner(){
 const wrap=document.getElementById("innerWrap"), agg=document.getElementById("innerAgg");
 if(!curOuter){ wrap.innerHTML="<div style='color:#888'>外側セルをクリック</div>"; agg.innerHTML=""; return; }
 const [a,b]=curOuter.split(",").map(Number);
 const ic=innerCells(curMk,a,b);
 const get=(i,j)=>{ const e=ic[i+","+j]; if(!e)return null; return {n:e.n, show:e.m}; };
 wrap.innerHTML=grid(`内側: bench HP — ${curMk} / 自分active=${L[a]} 相手active=${L[b]}`,get,null);
 let tot=0,ms=0,ts=0; for(const k in ic){const e=ic[k];tot+=e.n;ms+=e.m*e.n;ts+=e.t*e.n;}
 agg.innerHTML= tot? `この局面 集計 n=${tot} / model=${(ms/tot).toFixed(3)} / teacher=${(ts/tot).toFixed(3)}`:"データなし";
}
function renderMatchups(){
 const d=document.getElementById("matchups");
 d.innerHTML=DATA.matchups.map(mk=>`<button class="mk${mk===curMk?' active':''}" data-mk="${mk}">${mk}</button>`).join("");
 d.querySelectorAll("button").forEach(b=>{ b.onclick=()=>{ curMk=b.dataset.mk; curOuter=null;
  renderMatchups(); renderOuter(); renderInner(); }; });
}
document.getElementById("meta").textContent=
 `stage=${DATA.stage} / checkpoint=${DATA.checkpoint} / min_n=${MINN} / 総セル=${DATA.cells.length}`;
renderMatchups(); renderOuter(); renderInner();
</script></body></html>
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--stage", type=str, choices=["3a", "3b", "3c"], default="3c")
    p.add_argument("--out", type=Path, required=True, help="出力 HTML パス。")
    p.add_argument("--num-games", type=int, default=32)
    p.add_argument("--num-eval-games", type=int, default=4000)
    p.add_argument("--min-n", type=int, default=30)
    p.add_argument("--max-batch-size", type=int, default=None)
    p.add_argument("--trajectories-threshold", type=int, default=None)
    p.add_argument("--sleep-seconds", type=float, default=0.05)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--backend", type=str, choices=["local", "showdown"], default="local")
    p.add_argument("--random", dest="randomize", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--crit", dest="crit_enabled", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--sims", type=int, default=64)
    p.add_argument("--sim-concurrency", type=int, default=1)
    p.add_argument("--search-turn-min", type=int, default=6)
    p.add_argument("--search-turn-max", type=int, default=12)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    executor = get_rust_async_executor_wrapper()(
        args.num_games, args.max_batch_size, args.trajectories_threshold, args.backend,
        args.randomize, args.crit_enabled, args.stage, args.sims, args.sim_concurrency,
        args.search_turn_min, args.search_turn_max, False, False,
    )
    agent = Agent(device=args.device, checkpoint_path=args.checkpoint)
    print(f"checkpoint={args.checkpoint} stage={args.stage} num_eval_games={args.num_eval_games}")
    examples = collect_examples(executor, agent, args.num_eval_games, args.sleep_seconds)
    if not examples:
        print("交代局面サンプルが集まりませんでした。")
        return
    encoded = encode_observations(examples, agent.device)
    cells = hp_4d_switch_diagnostics(
        agent.model, encoded, examples, agent.device, agent.agent_config.amp_dtype
    )
    records = [
        [mk, a, b, c, d, round(m, 4), round(t, 4), n]
        for (mk, a, b, c, d), (m, t, n) in sorted(cells.items())
    ]
    present = [k for k in SWITCH_MATCHUP_KEYS if any(r[0] == k for r in records)]
    data = {
        "stage": args.stage,
        "checkpoint": str(args.checkpoint),
        "min_n": args.min_n,
        "labels": HP_BUCKET_LABELS,
        "matchups": present,
        "cells": records,
    }
    html = (
        _HTML_TEMPLATE
        .replace("__TITLE__", f"{args.stage} ({args.checkpoint.stem})")
        .replace("__DATA__", json.dumps(data, ensure_ascii=False))
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"総 item 数={len(examples)} / 4D セル数={len(records)} / 対面={present}")
    print(f"HTML を書き出しました: {args.out}")


if __name__ == "__main__":
    main()
