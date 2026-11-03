# CS2 Opening Scout

Learn where players go in the first 60 seconds of a CS2 round, like an opening book in chess.
Feed it demos, get a web report per map: where enemy Ts go (and how many), how often they rush,
enemy CT setups, and each player's and team's habits.

## What you get

`reports/opening_report.html`, a single page you open in a browser, filterable by map, enemies vs everyone, and buy type (pistol/eco/force/full):

- **As CT:** average enemies per site, a suggested 5-player split (e.g. `2 A · 1 Mid · 2 B`),
  how often ≥3 come to one site, rush rate (any T player on a site within 20 s, including 1–2 player rushes),
  T strategy per round (`A rush`, `B execute`, `Default > B`, ...), most common opening lines
  (`TSpawn > TYard > OutsideA > AStairs > AMain > BombsiteA`) and a per-player table.
- **As T:** enemy CT setups (`2A-1Mid-2B`), the thinnest site, each CT's usual position at 20 s.
- **Movement map** per map on the game's own radar image with callout names, filterable by side,
  time (first N seconds) and a single player.
- **Per team** and **per player** (Steam name, Steam ID, profile link), and **opening kills** (who and where).

"You" are auto-detected as the player present in the most demos; everyone on the other team is an enemy.

## Where the data comes from

**Your Steam account (recommended):**

```bash
python -m src.pipeline
```

First run shows a QR code: scan it in the Steam Mobile app (no password is typed or stored; the login
token is kept in `data/steam_refresh_token.txt`, delete it or run
`node tools/steam_gc/fetch_demo_urls.js --logout` to log out). The script asks the CS2 Game Coordinator
for your recent matches (the same list as Watch > Your Matches, max 8), downloads their demos from
Valve, also imports your CS2 replays folder, then builds the report with you as "me".
Valve keeps replays only for a limited time, so run it regularly to build up history.
Close CS2 while it runs (the account can only be in CS2 once).

Older matches: fill `data/sharecodes.txt` with `python -m src.data.steam_api --known-code CSGO-...`
(needs `STEAM_API_KEY`, `STEAM_ID`, `STEAM_AUTH_CODE` in `.env`); the next `src.pipeline` run asks the GC for those too.

**Other sources:**

| Source | How | Notes |
|---|---|---|
| CS2 replays folder | `collect --cs2-replays` | Demos you downloaded in-game. |
| HLTV | Download the GOTV demo, extract the .rar, `collect --from-dir` | Pro teams and players. |
| FACEIT | Data API → match → demo URL, `collect --urls` | Demo downloads need FACEIT Downloads API access (application, up to ~30 days). |

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
cd tools/steam_gc && npm install && cd ../..        # Node.js, for the Steam login

python -m src.data.collect --cs2-replays            # or --from-dir PATH / --urls urls.txt
python -m src.data.demo_parser                      # data/raw/demos -> data/parsed
python -m src.features.engineer                     # -> data/processed/{players,teams}.parquet
python -m src.data.radar                            # radar images from your CS2 install -> data/radar
python -m src.report.build_report                   # -> reports/opening_report.html (open #de_dust2 etc.)
```

Optional:

```bash
python -m src.features.engineer --list-places       # place names per map and the zone each maps to
python -m src.models.train                          # per-map route + position models -> models/<map>/
python -m src.api.server                            # POST /predict, GET /health
```

## How routes are labelled

- **Zone** of a position: a `BombsiteA`/`BombsiteB` place name; else the map's list in
  `configs/maps.yaml`; else geometry (clearly closer to one site's centre → that site, otherwise Mid).
- **T player route:** the first site reached within 60 s; if none, the last zone they were in.
  **Rush** = on a site within `rush_seconds` (20).
- **CT player route:** their zone at `ct_setup_seconds` (20).
- **T team strategy:** ≥3 rushers → `<site> rush`; ≥3 players to one site → `<site> execute`;
  otherwise `Default > <plant site>` or `Default`. 1–2 player rushes are counted separately.

Thresholds live in `configs/model_config.yaml`.

## Project structure

```
src/
  data/collect.py          gather demos (CS2 replays folder, any folder, URLs)
  data/steam_api.py        Steam sharecodes + sharecode decoder
  data/demo_parser.py      demo -> rounds / players / trajectory (1 sample per second, 60 s)
  data/radar.py            radar images + overview coordinates from the CS2 game files
  features/engineer.py     zones, routes, rushes, strategies, score/economy context
  features/schema.py       model features shared by training and the API
  report/                  HTML report builder + template
  models/                  XGBoost route classifier, MLP position predictor, train.py
  api/server.py            FastAPI
configs/                   model_config.yaml, maps.yaml
tests/
```

## Status

The report is the useful part today. The ML models train and serve, but with a handful of demos they do
no better than "always guess the most common route". They need many more demos per map.
