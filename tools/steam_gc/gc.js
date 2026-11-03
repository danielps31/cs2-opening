// Shared Steam pieces: login token storage and fetching your recent CS2 matches from the Game Coordinator.
//
// Files (all in data/, git-ignored):
//   steam_refresh_token.txt  login token (delete to log out)
//   steam_account.json       your steamid
//   steam_matches.json       recent matches (id, time, demo URL), merged with earlier runs
//   demo_urls.txt            one demo URL per line, for `python -m src.data.collect --urls`

const fs = require("fs");
const path = require("path");
const SteamUser = require("steam-user");
const GlobalOffensive = require("globaloffensive");

const DATA = path.resolve(__dirname, "../../data");
const TOKEN_FILE = path.join(DATA, "steam_refresh_token.txt");
const ACCOUNT_FILE = path.join(DATA, "steam_account.json");
const MATCHES_FILE = path.join(DATA, "steam_matches.json");
const URLS_FILE = path.join(DATA, "demo_urls.txt");
const SHARECODES_FILE = path.join(DATA, "sharecodes.txt");
const GC_TIMEOUT_MS = 60000;

function savedToken() {
  return fs.existsSync(TOKEN_FILE) ? fs.readFileSync(TOKEN_FILE, "utf8").trim() : null;
}

function saveToken(token) {
  fs.mkdirSync(DATA, {recursive: true});
  fs.writeFileSync(TOKEN_FILE, token);
}

function forgetToken() {
  if (fs.existsSync(TOKEN_FILE)) fs.unlinkSync(TOKEN_FILE);
}

function savedAccount() {
  return fs.existsSync(ACCOUNT_FILE) ? JSON.parse(fs.readFileSync(ACCOUNT_FILE, "utf8")) : null;
}

function demoUrl(match) {
  const rounds = match.roundstatsall || [];
  for (let i = rounds.length - 1; i >= 0; i--) {
    if (rounds[i].map && String(rounds[i].map).includes(".dem")) return rounds[i].map;
  }
  return null;
}

function saveMatches(matches) {
  const existing = fs.existsSync(MATCHES_FILE) ? JSON.parse(fs.readFileSync(MATCHES_FILE, "utf8")) : [];
  const byId = new Map(existing.map(m => [m.matchid, m]));
  for (const m of matches) {
    byId.set(String(m.matchid), {
      matchid: String(m.matchid),
      matchtime: m.matchtime ? new Date(m.matchtime * 1000).toISOString() : null,
      demo_url: demoUrl(m),
    });
  }
  const all = [...byId.values()].sort((a, b) => String(b.matchtime).localeCompare(String(a.matchtime)));
  fs.writeFileSync(MATCHES_FILE, JSON.stringify(all, null, 2));
  fs.writeFileSync(URLS_FILE, all.map(m => m.demo_url).filter(Boolean).join("\n") + "\n");
  return all;
}

// Log in with the saved token, ask the GC for recent matches (+ any saved sharecodes), save them.
// Resolves {steamid, name, received, total, withDemo}. An expired token is removed and rejects with code "EXPIRED".
function fetchMatches(refreshToken, log = () => {}) {
  return new Promise((resolve, reject) => {
    const user = new SteamUser();
    const csgo = new GlobalOffensive(user);
    const collected = [];
    let pending = 0, timer = null, done = false, name = null;

    const stop = (err, result) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      try { user.logOff(); } catch (e) {}
      err ? reject(err) : resolve(result);
    };
    const finish = () => {
      if (!collected.length) return stop(new Error("No answer from the CS2 Game Coordinator. Try again in a minute."));
      const all = saveMatches(collected);
      stop(null, {steamid: user.steamID.getSteamID64(), name, received: collected.length, total: all.length,
                  withDemo: all.filter(m => m.demo_url).length});
    };
    // Give up waiting (keeping whatever arrived) once the GC goes quiet
    const waitMore = () => { clearTimeout(timer); timer = setTimeout(finish, GC_TIMEOUT_MS); };

    user.on("error", err => {
      if (err.eresult === SteamUser.EResult.InvalidPassword || err.eresult === SteamUser.EResult.AccessDenied) {
        forgetToken();
        return stop(Object.assign(new Error("Saved login expired, log in again."), {code: "EXPIRED"}));
      }
      stop(new Error(`Steam error: ${err.message}`));
    });
    user.on("accountInfo", n => { name = n; });
    user.on("loggedOn", () => {
      const steamid = user.steamID.getSteamID64();
      fs.mkdirSync(DATA, {recursive: true});
      fs.writeFileSync(ACCOUNT_FILE, JSON.stringify({steamid}, null, 2));
      log(`Logged in as ${steamid}. Connecting to CS2...`);
      // Doesn't kick a CS2 session you have open elsewhere
      user.gamesPlayed([730]);
      waitMore();
    });
    user.on("playingState", blocked => {
      if (blocked) stop(Object.assign(new Error("CS2 is running on this account right now. Close CS2 and try again."), {code: "PLAYING"}));
    });
    csgo.on("connectedToGC", () => {
      // Older matches: sharecodes collected by `python -m src.data.steam_api`
      const codes = fs.existsSync(SHARECODES_FILE)
        ? fs.readFileSync(SHARECODES_FILE, "utf8").split(/\s+/).filter(c => c.startsWith("CSGO-"))
        : [];
      pending = 1 + codes.length;
      log(`Connected to CS2. Asking for recent matches${codes.length ? ` and ${codes.length} sharecodes` : ""}...`);
      csgo.on("matchList", matches => {
        collected.push(...matches);
        log(`${collected.length} matches received`);
        waitMore();
        if (--pending <= 0) finish();
      });
      csgo.requestRecentGames(user.steamID.getSteamID64());
      // The GC rate-limits requests, so space them out
      codes.forEach((code, i) => setTimeout(() => { csgo.requestGame(code); waitMore(); }, 1500 * (i + 1)));
    });
    user.logOn({refreshToken});
  });
}

module.exports = {DATA, URLS_FILE, savedToken, saveToken, forgetToken, savedAccount, fetchMatches};
