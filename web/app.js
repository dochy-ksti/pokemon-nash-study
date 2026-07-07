import init, { Battle } from "./pkg/poke_wasm.js";

// ページ名からステージを決める (battle-3c.html → "3c"、それ以外 → "3b")。
const STAGE = location.pathname.includes("battle-3c") ? "3c" : "3b";

// ---- i18n ----------------------------------------------------------------
const MON = {
  "Cloyster": { ja: "パルシェン", en: "Cloyster" },
  "Goodra-Hisui": { ja: "ヒスイヌメルゴン", en: "Goodra-Hisui" },
  "Goodra": { ja: "ヌメルゴン", en: "Goodra" },
};
const MOVE = {
  "shockwave": { ja: "でんげきは", en: "Shock Wave" },
  "bulldoze": { ja: "じならし", en: "Bulldoze" },
  "crunch": { ja: "かみくだく", en: "Crunch" },
  "darkpulse": { ja: "あくのはどう", en: "Dark Pulse" },
  "fightspe60": { ja: "とうしのは(仮)", en: "FightSpe60" },
  "fairyphy60": { ja: "ようせいのつめ(仮)", en: "FairyPhy60" },
};
const T = {
  tag: { ja: "ポケモンにおけるナッシュ均衡の研究", en: "A study of Nash equilibria in Pokémon" },
  title: { ja: "完全有利/不利対面 — ヒスイヌメルゴン vs パルシェン",
           en: "A hard-countered matchup — Goodra-Hisui vs Cloyster" },
  t_foe: { ja: "相手のポリシー", en: "Foe policy" },
  t_self: { ja: "自分のAI推奨手", en: "AI hint (you)" },
  foe_ai: { ja: "相手 AI", en: "Foe AI" },
  you: { ja: "あなた", en: "You" },
  pick_team: { ja: "あなたのチームを選ぶ", en: "Choose your team" },
  team_desc: { ja: "同じ2体でも技構成が異なります。相手AIは必ず反対の技構成で、先発はランダムです。",
               en: "Same two mons, different movesets. The foe AI always runs the opposite moveset; its lead is random." },
  pick_lead: { ja: "先発を選ぶ", en: "Choose your lead" },
  start: { ja: "対戦開始", en: "Start battle" },
  choose: { ja: "あなたの手を選ぶ（同時手番 — 決定と同時に相手の手も開示）",
            en: "Choose your move (simultaneous — both reveal at once)" },
  note: { ja: "推定ダメージは相手が交代しなかった場合の通常時 min–max（急所・確定数は非表示）",
          en: "Damage is the no-crit min–max when the foe stays in (crit / KO-count hidden)" },
  again: { ja: "もう一度", en: "Play again" },
  bench: { ja: "控え", en: "Bench" },
  foe_prob: { ja: "相手の着手確率", en: "Foe move probability" },
  ai_rec: { ja: "AI推奨（自分視点）", en: "AI hint (your view)" },
  switch_to: { ja: "へ交代", en: "switch" },
  rec: { ja: "推奨", en: "Best" },
  if_stay: { ja: "相手が居座った場合", en: "if foe stays in" },
  if_foe: { ja: "相手が", en: "If foe uses" },
  if_hits: { ja: "を撃つと あなたに", en: "→ you take" },
  win: { ja: "あなたの勝ち！", en: "You win!" },
  lose: { ja: "あなたの負け…", en: "You lose…" },
  draw: { ja: "引き分け", en: "Draw" },
  act_switch: { ja: "交代", en: "Switch" },
  crit: { ja: "急所！", en: "Crit!" },
  ko: { ja: "きぜつ！", en: "KO!" },
  foe_act_label: { ja: "相手の行動", en: "Foe's move" },
  you_act_label: { ja: "あなたの行動", en: "Your move" },
  win_rate: { ja: "AI評価によるあなたの勝率", en: "Your win rate (AI estimate)" },
  win_rate_foe: { ja: "相手AI評価による相手の勝率", en: "Foe's win rate (foe AI estimate)" },
  overview: { ja: "← 研究概要", en: "← Overview" },
};
// 3c はステージ固有の副題に差し替える (3b はデフォルトのまま)。
if (STAGE === "3c") {
  T.title = { ja: "真・極端な有利/不利対面 — ヌメルゴン vs パルシェン",
              en: "A truly extreme matchup — Goodra vs Cloyster" };
}
// URL の ?lang= で言語指定 (ja/en)。未指定・不正はデフォルト英語。
function langFromUrl() {
  return new URLSearchParams(location.search).get("lang") === "ja" ? "ja" : "en";
}
let lang = langFromUrl();
const monName = (raw) => (MON[raw]?.[lang]) ?? raw;
const moveName = (id) => (MOVE[id]?.[lang]) ?? id;
const tr = (k) => T[k][lang];

function applyStaticI18n() {
  document.documentElement.lang = lang;
  document.title = tr("tag");
  for (const el of document.querySelectorAll("[data-i18n]")) el.textContent = tr(el.dataset.i18n);
  // 研究概要への戻りリンクに現在の言語を引き継ぐ。
  const ov = document.getElementById("overview-link");
  if (ov) ov.href = "./" + (lang === "ja" ? "?lang=ja" : "");
}

// ---- state ---------------------------------------------------------------
let META, TABLE, VALUE_TABLE, H, PROB, VSCALE, SENT;
let battle = null, humanTeam = 0, humanLead = 0, aiTeam = 0, aiLead = 0;
let showFoe = true, showSelf = true, gameOver = false;
// 直前ターンの各サイドの手 {p1,p2:{kind,label,crit,ko}} と、ターン開始時の全メンバーHP。
let lastAct = null, preHp = null;
const el = (id) => document.getElementById(id);

// ---- policy table lookup -------------------------------------------------
function bucket(hp, maxHp) {
  if (hp <= 0) return 0;
  let k = Math.round((hp / maxHp) * (H - 1));
  return Math.min(Math.max(k, 1), H - 1); // 生存個体は 0 バケットに丸めない
}
// side.members は [Cloyster, Goodra] 固定順。active は party index。
function sideBuckets(side) {
  return {
    active: side.active,
    c: bucket(side.members[0].hp, side.members[0].max_hp),
    g: bucket(side.members[1].hp, side.members[1].max_hp),
  };
}
// クロスチーム限定: opp_team = 1 - ai_team なので次元に持たない。
function denseIndex(aiTeamId, ai, opp) {
  let k = aiTeamId;
  k = k * 2 + ai.active;
  k = k * H + ai.c;
  k = k * H + ai.g;
  k = k * 2 + opp.active;
  k = k * H + opp.c;
  k = k * H + opp.g;
  return k;
}
// aiSide の P(交代)。表に無い(番兵)なら null。相手は必ず反対チーム。
function pSwitch(aiSide, aiTeamId, oppSide) {
  const v = TABLE[denseIndex(aiTeamId, sideBuckets(aiSide), sideBuckets(oppSide))];
  return v === SENT ? null : v / PROB;
}
// aiSide 視点の勝率 (value head 0..1)。表に無いなら null。
function winRate(aiSide, aiTeamId, oppSide) {
  const v = VALUE_TABLE[denseIndex(aiTeamId, sideBuckets(aiSide), sideBuckets(oppSide))];
  return v === SENT ? null : v / VSCALE;
}
// 勝率帯を1つ描画。wv が null なら非表示。flip=true (相手側) は「あなた目線」で
// 色を反転し、勝率が高い=あなたに不利=赤 とする。
function renderWinrate(wr, wv, label, flip) {
  if (wv === null) { wr.classList.add("hidden"); return; }
  const good = flip ? wv < 0.5 : wv >= 0.5;
  const pct = (wv * 100).toFixed(1);
  wr.classList.remove("hidden");
  wr.classList.toggle("hi", good);
  wr.classList.toggle("lo", !good);
  wr.innerHTML = `<span class="wr-label">${label}</span><span class="wr-val">${pct}%</span>`;
}

// ---- rendering -----------------------------------------------------------
function hpClass(frac) { return frac > 0.5 ? "hp-hi" : frac > 0.2 ? "hp-mid" : "hp-lo"; }

function benchOf(side) {
  const i = side.active === 0 ? 1 : 0;
  return { mon: side.members[i], idx: i };
}

function monCardHtml(side, incomingHtml, lostPct) {
  const a = side.members[side.active];
  const frac = a.max_hp > 0 ? Math.max(0, a.hp) / a.max_hp : 0;
  const pct = Math.round(frac * 100);
  // 今ターン失った分を、現HPバーの右隣に赤セグメントとして並べる (次の手まで残る)。
  const loss = Math.max(0, Math.min(100 - pct, lostPct || 0));
  const lossHtml = loss > 0.05
    ? `<div class="hploss" style="width:${loss}%"></div>` : "";
  const types = a.types.map((t) => `<span class="type t-${t}">${t}</span>`).join("");
  const st = a.stats;
  const statCells = [["HP", st.hp], ["Atk", st.atk], ["Def", st.def],
    ["SpA", st.spa], ["SpD", st.spd], ["Spe", st.spe]]
    .map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
  const b = benchOf(side);
  const bfnt = b.mon.hp <= 0;
  const bpct = b.mon.max_hp > 0 ? Math.round(Math.max(0, b.mon.hp) / b.mon.max_hp * 100) : 0;
  return `<div class="id">
    <div class="name-row"><span class="mon-name">${monName(a.species)}</span><span class="types">${types}</span></div>
    <div class="hpline"><div class="hpbar">
        <div class="hpfill ${hpClass(frac)}" style="width:${pct}%"></div>${lossHtml}</div>
      <span class="hpnum"><b>${Math.max(0, a.hp)}</b>/${a.max_hp} · <b>${pct}%</b></span></div>
    <div class="stats">${statCells}</div>
    <div class="bench"><span class="dot ${bfnt ? "fnt" : ""}"></span>${tr("bench")} <b>${monName(b.mon.species)}</b>
      <span class="hpnum">${Math.max(0, b.mon.hp)}/${b.mon.max_hp} · ${bpct}%</span></div>
    ${incomingHtml || ""}
  </div>`;
}

function policyPanelHtml(kind, activeMove, pSw) {
  const cls = kind === "foe" ? "foe" : "self";
  const cap = kind === "foe" ? tr("foe_prob") : tr("ai_rec");
  if (pSw === null) return "";
  const pMove = 1 - pSw;
  const recMove = pMove >= pSw;
  const row = (label, p, rec) =>
    `<div class="prow ${rec ? "rec" : ""}"><div class="top-l"><span>${label}</span>
       <span class="pct">${(p * 100).toFixed(1)}%</span></div>
       <div class="track"><i style="width:${Math.round(p * 100)}%"></i></div></div>`;
  const b = kind; void b;
  return `<div class="policy ${cls}"><div class="cap">◈ ${cap}</div>
    ${row(activeMove.label, pMove, recMove)}
    ${row(activeMove.switchLabel, pSw, !recMove)}</div>`;
}

function activeMoveInfo(side) {
  const a = side.members[side.active];
  const mv = a.moves[0]; // 3b は 1 技
  const b = benchOf(side);
  return {
    slot: mv.slot,
    label: moveName(mv.id),
    switchLabel: `${monName(b.mon.species)} ${tr("switch_to")}`,
    benchIdx: b.idx,
    benchAlive: b.mon.hp > 0,
  };
}

// 現在アクティブが「ターン開始時の同じメンバーの HP」から失った割合 (%)。
// 交代・強制交代で場のポケモンが変わっても、そのメンバー自身の開始HPと比較するので正しい。
function ghostLost(side, preArr) {
  if (!preArr) return 0;
  const a = side.members[side.active];
  if (a.max_hp <= 0) return 0;
  const lost = preArr[side.active] - Math.max(0, a.hp);
  return Math.max(0, lost) / a.max_hp * 100;
}

// 各サイドの手バッジ HTML。act = {kind,label,crit,ko}、dmgPct = そのサイドが相手に与えた% 。
function badgeHtml(kind, act, dmgPct) {
  if (!act) return "";
  const label = kind === "foe" ? tr("foe_act_label") : tr("you_act_label");
  let res = "";
  if (act.kind === 0) {
    if (act.ko) res = `→ <b class="ko">${tr("ko")}</b>`;
    else if (dmgPct > 0) res = `→ <b>${Math.round(dmgPct)}%</b>`;
    if (act.crit) res += ` <span class="crit">${tr("crit")}</span>`;
  }
  return `<span class="who">${label}:</span><span class="what">${act.label}</span>` +
    (res ? `<span class="res">${res}</span>` : "");
}

function render() {
  const s = battle.snapshot();
  // 相手 = P2, あなた = P1
  const foeMv = activeMoveInfo(s.p2);
  const youMv = activeMoveInfo(s.p1);

  // 相手が撃った場合のダメージ (attacker=1)
  const foeDmg = battle.damageRange(1, foeMv.slot);
  const incoming = `<div class="incoming">${tr("if_foe")} <b>${foeMv.label}</b> ${tr("if_hits")}
     <span class="dmg-in">${foeDmg.min_pct.toFixed(0)}–${foeDmg.max_pct.toFixed(0)}%</span></div>`;

  const foeP = showFoe ? pSwitch(s.p2, aiTeam, s.p1) : null;
  const selfP = showSelf ? pSwitch(s.p1, humanTeam, s.p2) : null;

  // 勝率帯を上下2箇所に分離表示。
  //  相手勝率 (相手=P2 視点の value): 相手フィールド上・tog-foe 連動・あなた目線で色反転。
  //  自分勝率 (自分=P1 視点の value): 味方フィールド下・tog-self 連動。
  // 各値は表に無い(番兵)なら null → その帯だけ非表示。
  renderWinrate(el("winrate-foe"), showFoe ? winRate(s.p2, aiTeam, s.p1) : null,
    tr("win_rate_foe"), true);
  renderWinrate(el("winrate-self"), showSelf ? winRate(s.p1, humanTeam, s.p2) : null,
    tr("win_rate"), false);

  // ゴースト帯: 各サイドが今ターン失った割合。相手に与えた% は相手側のゴーストと一致。
  const ghFoe = ghostLost(s.p2, preHp?.p2); // 相手が失った = あなたが与えた
  const ghYou = ghostLost(s.p1, preHp?.p1); // あなたが失った = 相手が与えた

  // 手バッジ (場の上)。
  const foeBadge = el("foe-act"), youBadge = el("you-act");
  if (lastAct) {
    foeBadge.innerHTML = badgeHtml("foe", lastAct.p2, ghYou);
    youBadge.innerHTML = badgeHtml("you", lastAct.p1, ghFoe);
    foeBadge.classList.toggle("hidden", !lastAct.p2);
    youBadge.classList.toggle("hidden", !lastAct.p1);
  } else {
    foeBadge.classList.add("hidden");
    youBadge.classList.add("hidden");
  }

  el("foe-mon").innerHTML = monCardHtml(s.p2, incoming, ghFoe) +
    (showFoe ? policyPanelHtml("foe", foeMv, foeP) : "");
  el("you-mon").innerHTML = (showSelf ? policyPanelHtml("self", youMv, selfP) : "") +
    monCardHtml(s.p1, "", ghYou);

  renderActions(s.p1, youMv, selfP);
  if (s.done) finish(s);
}

function renderActions(you, youMv, selfP) {
  const youDmg = battle.damageRange(0, youMv.slot);
  const recBadge = (selfP !== null && (1 - selfP) >= selfP)
    ? `<span class="rec-badge">${tr("rec")} ${((1 - selfP) * 100).toFixed(1)}%</span>` : "";
  const a = you.members[you.active];
  const mtype = a.moves[0].move_type;
  let html = `<button class="act" data-kind="0" data-arg="${youMv.slot}">
      <div class="a-top"><span class="a-name">${youMv.label}</span>
        <span style="display:flex;gap:8px;align-items:center">${recBadge}<span class="a-type t-${mtype}">${mtype}</span></span></div>
      <div class="a-meta"><span>${tr("if_stay")} <span class="dmg">${youDmg.min_pct.toFixed(0)}–${youDmg.max_pct.toFixed(0)}%</span></span></div>
    </button>`;
  const b = benchOf(you);
  const recSw = (selfP !== null && selfP > (1 - selfP))
    ? `<span class="rec-badge">${tr("rec")} ${(selfP * 100).toFixed(1)}%</span>` : "";
  html += `<button class="act switch" data-kind="1" data-arg="${b.idx}" ${b.mon.hp <= 0 ? "disabled" : ""}>
      <div class="a-top"><span class="a-name">${monName(b.mon.species)} ${tr("switch_to")}</span>
        <span style="display:flex;gap:8px;align-items:center">${recSw}<span class="a-type">Switch</span></span></div>
      <div class="a-meta"><span>${b.mon.hp <= 0 ? "—" : "→"}</span></div>
    </button>`;
  el("acts").innerHTML = html;
  if (!gameOver) {
    for (const btn of el("acts").querySelectorAll(".act:not(:disabled)")) {
      btn.addEventListener("click", () => onHumanChoice(+btn.dataset.kind, +btn.dataset.arg));
    }
  }
}

// ---- turn flow -----------------------------------------------------------
function parseLegal(flat) {
  const out = [];
  for (let i = 0; i < flat.length; i += 2) out.push({ kind: flat[i], arg: flat[i + 1] });
  return out;
}
function sampleAi(s) {
  const foeMv = activeMoveInfo(s.p2);
  const legal = parseLegal(battle.legal(1));
  const canSwitch = legal.some((c) => c.kind === 1);
  const p = pSwitch(s.p2, aiTeam, s.p1);
  if (canSwitch && p !== null && Math.random() < p) {
    const sw = legal.find((c) => c.kind === 1);
    return { kind: 1, arg: sw.arg };
  }
  return { kind: 0, arg: foeMv.slot };
}

function onHumanChoice(kind, arg) {
  if (gameOver) return;
  const pre = battle.snapshot();
  const ai = sampleAi(pre);
  preHp = {
    p1: pre.p1.members.map((m) => Math.max(0, m.hp)),
    p2: pre.p2.members.map((m) => Math.max(0, m.hp)),
  };
  const res = battle.step(kind, arg, ai.kind, ai.arg, Math.random());
  lastAct = computeActions({ p1: { kind, arg }, p2: { kind: ai.kind, arg: ai.arg } }, res.events, pre);
  render();
}

// 各サイドの手を {kind,label,crit,ko} にまとめる。label は選んだ手 (技名 or「交代」)、
// crit/ko は events から (相手を急所/瀕死させたか)。ダメージ% はゴースト帯から render で付ける。
function computeActions(choices, events, pre) {
  const mk = (side, me, foe) => {
    const ch = choices[side];
    let label;
    if (ch.kind === 1) {
      label = tr("act_switch");
    } else {
      const mv = events.find((e) => "Move" in e && e.Move.user.player === me);
      label = mv ? moveName(mv.Move.move_id)
        : moveName(pre[side].members[pre[side].active].moves[0].id);
    }
    return {
      kind: ch.kind,
      label,
      crit: events.some((e) => "Crit" in e && e.Crit.target.player === foe),
      ko: events.some((e) => "Faint" in e && e.Faint.target.player === foe),
    };
  };
  return { p1: mk("p1", 1, 2), p2: mk("p2", 2, 1) };
}

function finish(s) {
  gameOver = true;
  for (const b of el("acts").querySelectorAll(".act")) b.disabled = true;
  const banner = el("banner");
  banner.classList.remove("hidden", "win", "lose");
  if (s.winner === 0) { banner.textContent = tr("win"); banner.classList.add("win"); }
  else if (s.winner === 1) { banner.textContent = tr("lose"); banner.classList.add("lose"); }
  else { banner.textContent = tr("draw"); }
  el("again-btn").classList.remove("hidden");
}

// ---- landing / setup -----------------------------------------------------
function teamMovesetLabel(teamId) {
  // 各チームの Cloyster / Goodra が持つ技を probe battle から読む。
  const probe = new Battle(STAGE, teamId, 0, teamId === 0 ? 1 : 0, 0);
  const s = probe.snapshot();
  const ms = s.p1.members.map((m) => `${monName(m.species)}=${moveName(m.moves[0].id)}`);
  probe.free();
  return ms.join(" / ");
}
function buildLanding() {
  const tc = el("team-choices"); tc.innerHTML = "";
  for (const t of [0, 1]) {
    const chip = document.createElement("button");
    chip.className = "chip" + (t === humanTeam ? " on" : "");
    chip.innerHTML = `<b>Team ${t + 1}</b><br><span class="desc">${teamMovesetLabel(t)}</span>`;
    chip.addEventListener("click", () => { humanTeam = t; buildLanding(); });
    tc.appendChild(chip);
  }
  const lc = el("lead-choices"); lc.innerHTML = "";
  const probe = new Battle(STAGE, humanTeam, 0, humanTeam === 0 ? 1 : 0, 0);
  const members = probe.snapshot().p1.members; probe.free();
  members.forEach((m, i) => {
    const chip = document.createElement("button");
    chip.className = "chip" + (i === humanLead ? " on" : "");
    chip.textContent = monName(m.species);
    chip.addEventListener("click", () => { humanLead = i; buildLanding(); });
    lc.appendChild(chip);
  });
}

function startBattle() {
  aiTeam = humanTeam === 0 ? 1 : 0; // クロスチーム限定: 相手は必ず反対の技構成
  aiLead = Math.random() < 0.5 ? 0 : 1;
  if (battle) battle.free();
  battle = new Battle(STAGE, humanTeam, humanLead, aiTeam, aiLead);
  gameOver = false;
  lastAct = null; preHp = null;
  el("foe-act").classList.add("hidden");
  el("you-act").classList.add("hidden");
  el("banner").classList.add("hidden");
  el("again-btn").classList.add("hidden");
  el("landing").classList.add("hidden");
  el("battle").classList.remove("hidden");
  el("foe-team").textContent = `Team ${aiTeam + 1}`;
  el("you-team").textContent = `Team ${humanTeam + 1}`;
  render();
}

// ---- boot ----------------------------------------------------------------
async function main() {
  await init();
  META = await (await fetch(`./policy_${STAGE}.meta.json`)).json();
  H = META.hp_buckets; PROB = META.prob_scale; SENT = META.sentinel;
  VSCALE = META.value_scale;
  const buf = await (await fetch(`./policy_${STAGE}.bin`)).arrayBuffer();
  TABLE = new Uint16Array(buf);
  const vbuf = await (await fetch(`./value_${STAGE}.bin`)).arrayBuffer();
  VALUE_TABLE = new Uint16Array(vbuf);

  applyStaticI18n();
  buildLanding();

  el("start-btn").addEventListener("click", startBattle);
  el("again-btn").addEventListener("click", () => {
    el("battle").classList.add("hidden");
    el("landing").classList.remove("hidden");
  });
  el("tog-foe").addEventListener("click", () => {
    showFoe = !showFoe; el("tog-foe").classList.toggle("on", showFoe);
    if (battle && !el("battle").classList.contains("hidden")) render();
  });
  el("tog-self").addEventListener("click", () => {
    showSelf = !showSelf; el("tog-self").classList.toggle("on", showSelf);
    if (battle && !el("battle").classList.contains("hidden")) render();
  });
  for (const b of document.querySelectorAll(".lg")) {
    // 起動時: URL 由来の lang に合わせてボタンの on 状態を反映。
    b.classList.toggle("on", b.dataset.lang === lang);
    b.addEventListener("click", () => {
      lang = b.dataset.lang;
      for (const x of document.querySelectorAll(".lg")) x.classList.toggle("on", x === b);
      // 共有できるよう URL に ?lang= を反映 (履歴は増やさない)。
      const u = new URL(location.href);
      u.searchParams.set("lang", lang);
      history.replaceState(null, "", u);
      applyStaticI18n(); buildLanding();
      if (battle && !el("battle").classList.contains("hidden")) render();
    });
  }
}
main();
