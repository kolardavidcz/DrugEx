const fs = require("fs");
const cp = require("child_process");

globalThis.DATABASE = {};
const output = cp.execSync("git ls-tree --name-only origin/education docs/data/").toString().trim();
const files = output.split("\n");

for (const file of files) {
  if (file.endsWith(".js") && !file.includes("sections")) {
    const code = cp.execSync("git show origin/education:" + file).toString();
    eval(code);
  }
}

console.log("Loaded symbols from origin/education:", Object.keys(globalThis.DATABASE).length);
fs.writeFileSync("/tmp/drugex_education_db.json", JSON.stringify(globalThis.DATABASE, null, 2));
