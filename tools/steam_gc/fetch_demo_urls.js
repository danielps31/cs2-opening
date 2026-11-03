// Fetch demo URLs of your recent CS2 matches from the Game Coordinator (terminal version).
//
//   node tools/steam_gc/fetch_demo_urls.js            # first run: scan a QR code with the Steam Mobile app
//   node tools/steam_gc/fetch_demo_urls.js --logout   # forget the saved login
//
// For a browser login page instead, run `python -m src.app`.

const {LoginSession, EAuthTokenPlatformType} = require("steam-session");
const qrcode = require("qrcode-terminal");
const gc = require("./gc");

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

async function main() {
  if (process.argv.includes("--logout")) {
    gc.forgetToken();
    console.log("Saved Steam login removed.");
    return;
  }
  let token = gc.savedToken();
  if (!token) {
    token = await loginWithQR();
    gc.saveToken(token);
  }
  const r = await gc.fetchMatches(token, msg => console.log(msg));
  console.log(`${r.received} matches received, ${r.total} known in total, ${r.withDemo} with a demo link.`);
  console.log(`Demo links: ${gc.URLS_FILE}`);
  setTimeout(() => process.exit(0), 500);
}

main().catch(err => fail(err.message));
