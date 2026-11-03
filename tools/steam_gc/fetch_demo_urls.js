// Fetch demo URLs of your recent CS2 matches from the Game Coordinator.
//
//   node tools/steam_gc/fetch_demo_urls.js            # first run: scan a QR code with the Steam Mobile app
//   node tools/steam_gc/fetch_demo_urls.js --logout   # forget the saved login
//
// Your password is never asked for or stored. The login token stays in
// data/steam_refresh_token.txt (git-ignored); delete it to log out.
//
// Writes:
//   data/steam_account.json   your steamid
//   data/steam_matches.json   recent matches (id, time, demo URL), merged with earlier runs
//   data/demo_urls.txt        one demo URL per line, for `python -m src.data.collect --urls`

const fs = require("fs");
const path = require("path");
const SteamUser = require("steam-user");
const GlobalOffensive = require("globaloffensive");
const {LoginSession, EAuthTokenPlatformType} = require("steam-session");
const qrcode = require("qrcode-terminal");

const DATA = path.resolve(__dirname, "../../data");
const TOKEN_FILE = path.join(DATA, "steam_refresh_token.txt");
const ACCOUNT_FILE = path.join(DATA, "steam_account.json");
const MATCHES_FILE = path.join(DATA, "steam_matches.json");
const URLS_FILE = path.join(DATA, "demo_urls.txt");
const SHARECODES_FILE = path.join(DATA, "sharecodes.txt");
const GC_TIMEOUT_MS = 60000;

function fail(message) {
  console.error(message);
  process.exit(1);
}

async function loginWithQR() {
  const session = new LoginSession(EAuthTokenPlatformType.SteamClient);
  const {qrChallengeUrl} = await session.startWithQR();
  console.log("\nScan this QR code with the Steam Mobile app (Steam Guard tab > scan):\n");
  qrcode.generate(qrChallengeUrl, {small: true});
  return new Promise((resolve, reject) => {
    session.on("remoteInteraction", () => console.log("QR scanned, confirm the login in the app..."));
    session.on("authenticated", () => resolve(session.refreshToken));
    session.on("timeout", () => reject(new Error("QR login timed out, run the script again")));
    session.on("error", reject);
  });
}

async function refreshToken() {
  if (fs.existsSync(TOKEN_FILE)) return fs.readFileSync(TOKEN_FILE, "utf8").trim();
  const token = await loginWithQR();
  fs.mkdirSync(DATA, {recursive: true});
  fs.writeFileSync(TOKEN_FILE, token);
  return token;
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
  const urls = all.map(m => m.demo_url).filter(Boolean);
  fs.writeFileSync(URLS_FILE, urls.join("\n") + "\n");
  return all;
}

async function main() {
  if (process.argv.includes("--logout")) {
    if (fs.existsSync(TOKEN_FILE)) fs.unlinkSync(TOKEN_FILE);
    console.log("Saved Steam login removed.");
    return;
  }

  const user = new SteamUser();
  const csgo = new GlobalOffensive(user);
  user.logOn({refreshToken: await refreshToken()});

  user.on("error", err => {
    if (err.eresult === SteamUser.EResult.InvalidPassword || err.eresult === SteamUser.EResult.AccessDenied) {
      fs.unlinkSync(TOKEN_FILE);
      fail("Saved login expired. Run the script again to scan a new QR code.");
    }
    fail(`Steam error: ${err.message}`);
  });

  user.on("loggedOn", () => {
    const steamid = user.steamID.getSteamID64();
    fs.mkdirSync(DATA, {recursive: true});
    fs.writeFileSync(ACCOUNT_FILE, JSON.stringify({steamid}, null, 2));
    console.log(`Logged in as ${steamid}. Connecting to CS2...`);
    // Don't kick a CS2 session you have open elsewhere
    user.gamesPlayed([730]);
  });

  user.on("playingState", blocked => {
    if (blocked) fail("CS2 is running on this account right now. Close CS2 and run the script again.");
  });

  const collected = [];
  let pending = 0;
  let timer = null;
  const finish = () => {
    clearTimeout(timer);
    if (!collected.length) fail("No answer from the CS2 Game Coordinator. Try again in a minute.");
    const all = saveMatches(collected);
    const withDemo = all.filter(m => m.demo_url).length;
    console.log(`${collected.length} matches received, ${all.length} known in total, ${withDemo} with a demo link.`);
    console.log(`Demo links: ${URLS_FILE}`);
    user.logOff();
    setTimeout(() => process.exit(0), 500);
  };
  // Give up waiting (keeping whatever arrived) once the GC goes quiet
  const waitMore = () => { clearTimeout(timer); timer = setTimeout(finish, GC_TIMEOUT_MS); };
  waitMore();

  csgo.on("connectedToGC", () => {
    // Older matches: sharecodes collected by `python -m src.data.steam_api`
    const codes = fs.existsSync(SHARECODES_FILE)
      ? fs.readFileSync(SHARECODES_FILE, "utf8").split(/\s+/).filter(c => c.startsWith("CSGO-"))
      : [];
    pending = 1 + codes.length;
    csgo.on("matchList", matches => {
      collected.push(...matches);
      waitMore();
      if (--pending <= 0) finish();
    });
    csgo.requestRecentGames(user.steamID.getSteamID64());
    // The GC rate-limits requests, so space them out
    codes.forEach((code, i) => setTimeout(() => { csgo.requestGame(code); waitMore(); }, 1500 * (i + 1)));
  });
}

main().catch(err => fail(err.message));
