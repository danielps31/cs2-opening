import pandas as pd

from src.data.steam_api import decode_sharecode
from src.features.engineer import ZoneMapper, opening_line, round_context, team_strategy


def make_samples():
    return pd.DataFrame({
        "last_place_name": ["BombsiteA", "BombsiteA", "BombsiteB", "BombsiteB"],
        "X": [0.0, 10.0, 1000.0, 1010.0],
        "Y": [0.0, 10.0, 0.0, 10.0],
    })


def test_zone_mapper_uses_site_names_config_then_geometry():
    zm = ZoneMapper("de_dust2", make_samples())
    assert zm.zone("BombsiteB", 0, 0) == "B"
    assert zm.zone("LongDoors", 1000, 0) == "A"   # config wins over position
    assert zm.zone("SomewhereNew", 50, 0) == "A"  # clearly closer to A
    assert zm.zone("SomewhereNew", 500, 0) == "Mid"
    assert zm.zone("TSpawn", 0, 0) is None


def test_opening_line_dedupes_and_keeps_destination():
    assert opening_line(["TSpawn", "TSpawn", "Middle", "", "BombsiteA"]) == "TSpawn > Middle > BombsiteA"
    long = [f"P{i}" for i in range(10)]
    assert opening_line(long, max_steps=4) == "P0 > P1 > P2 > P9"


def test_team_strategy_labels():
    t = lambda routes, rush: pd.DataFrame({"side": "T", "route": routes, "rush": rush})
    assert team_strategy(t(["B"] * 4 + ["A"], [1, 1, 1, 0, 0]), "B")["strategy"] == "B rush"
    small = team_strategy(t(["B", "B", "A", "A", "Mid"], [1, 0, 0, 0, 0]), None)
    assert small["strategy"] == "Default" and small["rushers"] == 1 and small["rush_site"] == "B"
    assert team_strategy(t(["A", "A", "A", "B", "Mid"], [0] * 5), "A")["strategy"] == "A execute"
    ct = pd.DataFrame({"side": "CT", "route": ["A", "A", "Mid", "B", "B"], "rush": 0})
    assert team_strategy(ct, None)["strategy"] == "2A-1Mid-2B"


def test_round_context_tracks_score_across_halftime():
    players = pd.DataFrame(
        [(r, f"p{i}", ("T" if i < 5 else "CT") if r <= 12 else ("CT" if i < 5 else "T"))
         for r in range(1, 15) for i in range(10)],
        columns=["round_number", "steamid", "side"])
    # Team 0 (T first half) wins rounds 1-12, loses 13-14 as CT
    rounds = pd.DataFrame({"round_number": range(1, 15),
                           "winner": ["T"] * 12 + ["T", "T"]})
    ctx = round_context(rounds, players).set_index(["round_number", "side"])
    assert ctx.loc[(13, "CT"), "score_team"] == 12      # team 0 is CT now
    assert ctx.loc[(13, "T"), "score_team"] == 0
    assert ctx.loc[(13, "T"), "loss_bonus"] == 1400     # reset at halftime
    assert ctx.loc[(14, "CT"), "prev_round_won"] == 0


def test_decode_sharecode_known_example():
    d = decode_sharecode("CSGO-GADqf-jjyJ8-cSP2r-smZRo-TO2xK")
    assert d == {"match_id": 3230642215713767580, "outcome_id": 3230647599455273103, "token": 55788}
