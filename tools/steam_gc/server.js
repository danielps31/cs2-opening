// Local web app: Steam login page + the opening report.
//
//   node tools/steam_gc/server.js [port]      (or: python -m src.app)
//
// Log in with a QR code (Steam Mobile app) or account name + password + Steam Guard. The password goes
// straight to Steam and is never stored; only the login token is kept in data/steam_refresh_token.txt.
// After login it fetches your recent matches from the CS2 Game Coordinator, downloads the demos and
// rebuilds the report. Listens on 127.0.0.1 only.

const fs = require("fs");
const http = require("http");
const path = require("path");
const {spawn} = require("child_process");
const QRCode = require("qrcode");
const {LoginSession, EAuthTokenPlatformType, EAuthSessionGuardType} = require("steam-session");
const gc = require("./gc");

const ROOT = path.resolve(__dirname, "../..");
const REPORT = path.join(ROOT, "reports", "opening_report.html");
const LOGIN_PAGE = path.join(__dirname, "login.html");
const PORT = +process.argv[2] || 8765;
const VENV_PY = path.join(ROOT, ".venv", "Scripts", "python.exe");
const PYTHON = fs.existsSync(VENV_PY) ? VENV_PY : "python";

// ---- state shown on the login page ------------------------------------------------------
let session = null;
let login = {state: "idle"};       // idle | qr | scanned | code | confirm | done | error
let job = {state: "idle", log: []}; // idle | running | done | error

const jobLog = msg => {
  job.log.push(`${new Date().toLocaleTimeString()}  ${msg}`);
  if (job.log.length > 300) job.log.shift();
};

function newSession() {
  if (session) session.cancelLoginAttempt().catch(() => {});
  session = new LoginSession(EAuthTokenPlatformType.SteamClient);
  session.on("remoteInteraction", () => { if (login.state === "qr") login = {...login, state: "scanned"}; });
  session.on("authenticated", () => {
    gc.saveToken(session.refreshToken);
    login = {state: "done"};
    sync();
  });
  session.on("timeout", () => { login = {state: "error", error: "timeout"}; });
  session.on("error", err => { login = {state: "error", error: err.message}; });
  return session;
}

// ---- fetch matches, download demos, rebuild the report ------------------------------------
function run(cmd, args) {
  return new Promise((resolve, reject) => {
    jobLog(`$ ${path.basename(cmd)} ${args.join(" ")}`);
    const p = spawn(cmd, args, {cwd: ROOT, env: {...process.env, PYTHONIOENCODING: "utf-8"}});
    const onData = d => String(d).split(/\r?\n/).filter(Boolean).forEach(l => jobLog(l.slice(0, 300)));
    p.stdout.on("data", onData);
    p.stderr.on("data", onData);
    p.on("error", reject);
    p.on("close", code => code === 0 ? resolve() : reject(new Error(`${path.basename(cmd)} exited with ${code}`)));
  });
}

async function sync() {
  if (job.state === "running") return;
  const token = gc.savedToken();
  if (!token) return;
  job = {state: "running", log: []};
  try {
    const r = await gc.fetchMatches(token, jobLog);
    jobLog(`${r.received} matches from Steam, ${r.withDemo} with a demo link`);
    await run(PYTHON, ["-m", "src.pipeline", "--no-steam", "--urls", gc.URLS_FILE, "--me", r.steamid]);
    job.state = "done";
    jobLog("Report ready.");
  } catch (err) {
    job.state = "error";
    jobLog(`Error: ${err.message}`);
    if (err.code === "EXPIRED") login = {state: "idle"};
  }
}

// ---- HTTP ---------------------------------------------------------------------------------
const send = (res, status, body, type = "application/json") => {
  res.writeHead(status, {"Content-Type": type + "; charset=utf-8", "Cache-Control": "no-store"});
  res.end(typeof body === "string" ? body : JSON.stringify(body));
};

const readJson = req => new Promise(resolve => {
  let data = "";
  req.on("data", c => { data += c; if (data.length > 1e5) req.destroy(); });
  req.on("end", () => { try { resolve(JSON.parse(data || "{}")); } catch (e) { resolve({}); } });
});

const routes = {
  "GET /api/status": async () => ({
    loggedIn: !!gc.savedToken(), account: gc.savedAccount(), login: {...login}, job,
    reportReady: fs.existsSync(REPORT),
  }),
  "POST /api/login/qr": async () => {
    const s = newSession();
    const {qrChallengeUrl} = await s.startWithQR();
    login = {state: "qr", qr: await QRCode.toDataURL(qrChallengeUrl, {margin: 1, width: 260})};
    return {ok: true};
  },
  "POST /api/login/password": async body => {
    if (!body.username || !body.password) return {ok: false, error: "missing"};
    const s = newSession();
    try {
      const r = await s.startWithCredentials({accountName: body.username, password: body.password});
      if (r.actionRequired) {
        const types = r.validActions.map(a => a.type);
        const email = r.validActions.find(a => a.type === EAuthSessionGuardType.EmailCode);
        if (types.includes(EAuthSessionGuardType.DeviceCode)) login = {state: "code", codeType: "device"};
        else if (email) login = {state: "code", codeType: "email", detail: email.detail};
        else login = {state: "confirm"};
      }
      return {ok: true};
    } catch (err) {
      login = {state: "error", error: err.message};
      return {ok: false, error: err.message};
    }
  },
  "POST /api/login/code": async body => {
    try {
      await session.submitSteamGuardCode(String(body.code || "").trim());
      return {ok: true};
    } catch (err) {
      return {ok: false, error: err.message};
    }
  },
  "POST /api/logout": async () => {
    gc.forgetToken();
    login = {state: "idle"};
    return {ok: true};
  },
  "POST /api/sync": async () => { sync(); return {ok: true}; },
};

const server = http.createServer(async (req, res) => {
  // Only this machine, and no requests forged by other websites
  const host = (req.headers.host || "").replace(/:\d+$/, "");
  if (!["127.0.0.1", "localhost"].includes(host)) return send(res, 403, {error: "forbidden"});
  const url = new URL(req.url, `http://${req.headers.host}`);
  if (req.method === "POST" && req.headers["x-cs2-opening"] !== "1") return send(res, 403, {error: "forbidden"});

  const route = routes[`${req.method} ${url.pathname}`];
  if (route) {
    try { return send(res, 200, await route(req.method === "POST" ? await readJson(req) : {})); }
    catch (err) { return send(res, 500, {ok: false, error: err.message}); }
  }
  if (req.method === "GET" && url.pathname === "/login") return send(res, 200, fs.readFileSync(LOGIN_PAGE, "utf8"), "text/html");
  if (req.method === "GET" && url.pathname === "/") {
    if (!fs.existsSync(REPORT)) { res.writeHead(302, {Location: "/login"}); return res.end(); }
    return send(res, 200, fs.readFileSync(REPORT, "utf8"), "text/html");
  }
  send(res, 404, {error: "not found"});
});

server.listen(PORT, "127.0.0.1", () => console.log(`CS2 Opening: http://127.0.0.1:${PORT}/login`));
