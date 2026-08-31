/**
 * DrugEx Hub — Static Build & Bundle Synchronizer
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, "..");
const publicDir = path.join(rootDir, "public");

function copyDirRecursive(src, dest) {
  if (!fs.existsSync(dest)) {
    fs.mkdirSync(dest, { recursive: true });
  }

  const entries = fs.readdirSync(src, { withFileTypes: true });
  for (const entry of entries) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);

    if (entry.isDirectory()) {
      copyDirRecursive(srcPath, destPath);
    } else {
      fs.copyFileSync(srcPath, destPath);
    }
  }
}

console.log("=================================================");
console.log(" 📦 DrugEx Hub — Static Bundle Synchronizer ");
console.log("=================================================");

copyDirRecursive(path.join(rootDir, "app"), path.join(publicDir, "app"));
copyDirRecursive(path.join(rootDir, "data"), path.join(publicDir, "data"));

// Copy index.html to public root for clean hosting
fs.copyFileSync(path.join(rootDir, "app/index.html"), path.join(publicDir, "index.html"));

console.log("✨ Synchronized app/ and data/ into public/ bundle successfully!");
