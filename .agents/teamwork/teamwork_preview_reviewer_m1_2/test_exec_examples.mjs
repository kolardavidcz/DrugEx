import fs from 'node:fs';
import vm from 'node:vm';
import { execSync } from 'node:child_process';

const html = fs.readFileSync('/home/kolar/build_projects/drugex_edu/docs/index.html', 'utf8');
const scriptMatch = html.match(/<script type="text\/babel">([\s\S]*?)<\/script>/);
const scriptCode = scriptMatch[1];
const dbStart = scriptCode.indexOf('const DATABASE = {');
const appStart = scriptCode.indexOf('function App()');
const dbCode = scriptCode.slice(dbStart, appStart).trim().replace(/^const DATABASE =/, 'globalThis.DATABASE =');

const context = {};
vm.createContext(context);
vm.runInContext(dbCode, context);
const db = context.DATABASE;

const testKeys = [
  'drugex::data::corpus::vocabulary::VocSmiles',
  'drugex::data::corpus::vocabulary::VocGraph',
  'drugex::molecules::mol::DrExMol',
  'drugex::parallel::ParallelProcessor',
  'drugex::molecules::converters::DefaultStandardizer',
  'drugex::data::fragments::Fragmenter'
];

console.log('Testing execution of selected sample examples in Python...');

for (const sym of testKeys) {
  const item = db[sym];
  console.log(`\n--- Testing ${sym} ---`);
  const code = item.example.code;
  const tempFile = '/home/kolar/build_projects/drugex_edu/.agents/teamwork/teamwork_preview_reviewer_m1_2/test_run.py';
  fs.writeFileSync(tempFile, code, 'utf8');

  try {
    const stdout = execSync(`python3 ${tempFile}`, {
      env: { ...process.env, PYTHONPATH: '/home/kolar/build_projects/drugex_edu/repo' },
      encoding: 'utf8',
      stdio: ['pipe', 'pipe', 'pipe']
    });
    console.log('[RUN SUCCESS]');
    console.log('ACTUAL OUTPUT:');
    console.log(stdout.trim());
    console.log('DOCUMENTED OUTPUT:');
    console.log(item.example.output.trim());
  } catch (err) {
    console.log('[RUN FAILED]');
    console.log(err.stderr ? err.stderr.toString() : err.message);
  }
}

if (fs.existsSync('/home/kolar/build_projects/drugex_edu/.agents/teamwork/teamwork_preview_reviewer_m1_2/test_run.py')) {
  fs.unlinkSync('/home/kolar/build_projects/drugex_edu/.agents/teamwork/teamwork_preview_reviewer_m1_2/test_run.py');
}
