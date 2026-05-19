import { existsSync, readFileSync } from "fs";
import path from "path";

let loaded = false;

function parseEnvLine(line: string) {
  const trimmed = line.trim();
  if (!trimmed || trimmed.startsWith("#")) return;

  const equalIndex = trimmed.indexOf("=");
  if (equalIndex <= 0) return;

  const key = trimmed.slice(0, equalIndex).trim();
  if (!key || process.env[key] !== undefined) return;

  let value = trimmed.slice(equalIndex + 1).trim();
  if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
    value = value.slice(1, -1);
  }
  process.env[key] = value;
}

function findRootEnvPath() {
  const candidates = [
    path.join(process.cwd(), ".env"),
    path.join(process.cwd(), "..", ".env"),
  ];

  for (const candidate of candidates) {
    if (existsSync(candidate)) return candidate;
  }

  return null;
}

export function ensureRootEnvLoaded() {
  if (loaded) return;
  loaded = true;

  const rootEnvPath = findRootEnvPath();
  if (!rootEnvPath) return;

  const content = readFileSync(rootEnvPath, "utf8");
  for (const line of content.split(/\r?\n/)) {
    parseEnvLine(line);
  }
}
