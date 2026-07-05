# A Study of Nash Equilibria in Pokémon

*日本語版は [readme-ja.md](readme-ja.md) を参照。*

This repository builds a Pokémon battle AI to investigate where the game's Nash
equilibria actually lie. There's no point keeping the findings to myself, so I'm
sharing them — and the learned policies are playable in your browser, GPU-free.

- **Overview & battle portal**: https://pokemon-nash-study.pages.dev/
- Every page has a JA/EN toggle in the top-right (`?lang=ja` / `?lang=en`; English
  is the default, and the language carries across links).

## Cycling in an extreme matchup — Goodra-Hisui vs Cloyster (Stage 3b)

[Source code](https://github.com/dochy-ksti/pokemon-nash-study)

**Goal.** Build an extreme favorable/unfavorable matchup and see whether the
decision collapses to a single option — always switch, or always attack.

**Setup.** I paired a Special-Defense-focused Hisuian Goodra with a
Defense-focused Cloyster and tuned their movesets to create a perfectly
favorable/unfavorable matchup for the AI to learn. The two share identical Attack
and Special Attack, and every real stat is identical except Defense and Special
Defense, which are swapped between them. Shock Wave (Special / Electric / 60 BP)
hits Cloyster for super-effective damage; Bulldoze (Physical / Ground / 60 BP)
hits Hisuian Goodra super-effectively — but each does very little to the other
Pokémon, so switching sharply cuts the damage taken. Movesets are mirrored across
the two teams, so for both sides "switching out of a bad matchup gives you a good
one," making cycling battles likely. Both Pokémon hold a Covert Cloak, which
suppresses the added (secondary) effects of moves. I trained the AI on this and
studied the strategy it converged to.

[▶ Play this battle](https://pokemon-nash-study.pages.dev/battle-3b.html)

**Result.** Even in a seemingly clear-cut matchup, surprisingly complex mind
games emerged. The tricky part: constrained by real moves, I couldn't build a
perfectly symmetric relationship — Bulldoze is neutral (1×) against Cloyster
while Shock Wave is resisted (½×) by Hisuian Goodra. As a result, "even in a
favorable matchup, don't fire Shock Wave — bait a switch and hit Bulldoze
instead" became a very strong line.

## A truly extreme matchup — Goodra vs Cloyster (Stage 3c)

[Source code](https://github.com/dochy-ksti/pokemon-nash-study)

**Goal.** Build a *truly* extreme favorable/unfavorable matchup and test whether
the decision collapses to a single option.

**Setup.** I paired a Special-Defense-focused Goodra with a Defense-focused
Cloyster, then gave them fictional moves — a 60-BP physical Fairy move and a
60-BP special Fighting move. Real stats are identical and the type-effectiveness
relationships are made perfectly symmetric, to see what happens.

[▶ Play this battle](https://pokemon-nash-study.pages.dev/battle-3c.html)

**Result.** It became considerably simpler, yet complex mind games still emerged.
Because you gain an edge by knocking out just one of the opponent's two Pokémon —
thereby denying them a switch — bait-switching to focus-fire a single target was
effective. In the end, a favorable matchup did not collapse to "always attack,"
nor an unfavorable one to "always switch."

## How it works

The battle demos run a Rust battle simulator (poke-sho-rust) compiled to
WebAssembly to resolve turns in the browser, while the AI's move comes from a
lookup into a policy table pre-computed over every state. No server and no GPU are
needed, so the whole thing runs on static hosting (Cloudflare Pages) alone.

## License

This project is released under the [MIT License](LICENSE).
