/**
 * DrugEx Hub — Automated Publication-Grade PDF Generator
 */

import puppeteer from 'puppeteer-core';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';
import http from 'http';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');
const outputPath = path.join(projectRoot, 'DrugEx_Full_Handbook.pdf');

// Potential Chrome executable paths on Windows / WSL
const CHROME_PATHS = [
  '/mnt/c/Program Files/Google/Chrome/Application/chrome.exe',
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  '/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe',
  'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium-browser',
  '/usr/bin/chromium'
];

function checkServer(url) {
  return new Promise((resolve) => {
    const req = http.get(url, (res) => {
      resolve(res.statusCode >= 200 && res.statusCode < 400);
    });
    req.on('error', () => resolve(false));
    req.setTimeout(1000, () => {
      req.destroy();
      resolve(false);
    });
  });
}

async function main() {
  console.log('=================================================');
  console.log(' 📚 DrugEx Hub — Publication PDF Generator ');
  console.log('=================================================');

  let serverProc = null;
  const isRunning = await checkServer('http://127.0.0.1:34100/app/index.html');
  if (!isRunning) {
    console.log('⚡ Dev server not active on port 34100. Spawning python3 serve.py 34100...');
    serverProc = spawn('python3', ['serve.py', '34100'], {
      cwd: projectRoot,
      stdio: 'ignore'
    });
    for (let i = 0; i < 25; i++) {
      await new Promise(r => setTimeout(r, 200));
      if (await checkServer('http://127.0.0.1:34100/app/index.html')) {
        console.log('✓ Dev server ready.');
        break;
      }
    }
  }

  try {
    let executablePath = CHROME_PATHS.find(p => fs.existsSync(p));
    if (!executablePath) {
      console.error('❌ Chrome executable not found in standard paths.');
      process.exit(1);
    }
    console.log(`✓ Using Chrome binary: ${executablePath}`);

    console.log('🚀 Launching headless browser...');
    const browser = await puppeteer.launch({
      executablePath,
      headless: true,
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-gpu',
        '--disable-dev-shm-usage',
        '--run-all-compositor-stages-before-draw'
      ]
    });

    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 900 });

    const url = 'http://127.0.0.1:34100/app/index.html#/handbook-print';
    console.log(`🌐 Navigating to ${url}...`);
    await page.goto(url, { waitUntil: 'networkidle0', timeout: 30000 });

  console.log('⏳ Waiting for full curriculum content and KaTeX rendering to settle...');
  await page.waitForSelector('.handbook-print-container', { timeout: 15000 });
  await page.waitForSelector('.slide-card', { timeout: 15000 });

  // Give KaTeX and MathJax 2 seconds to complete any pending typesetting
  await new Promise(r => setTimeout(r, 2000));

  const slideCount = await page.evaluate(() => {
    return document.querySelectorAll('.slide-card').length;
  });
  const quizCount = await page.evaluate(() => {
    return document.querySelectorAll('.quiz-qcard-compact').length;
  });
  console.log(`✓ Detected ${slideCount} rendered handbook slide cards and ${quizCount} compact quiz questions across all 6 modules.`);

  console.log('📄 Exporting continuous publication A4 PDF with academic headers & footers...');
  await page.pdf({
    path: outputPath,
    format: 'A4',
    printBackground: true,
    displayHeaderFooter: true,
    headerTemplate: `
      <div style="font-size: 7.5pt; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: #64748b; width: 100%; display: flex; justify-content: space-between; align-items: center; padding: 0 12mm; border-bottom: 0.5pt solid #e2e8f0;">
        <span style="font-weight: 600; color: #0284c7;">DrugEx Hub · De Novo Drug Design & ROCS Shape-Matching</span>
        <span>VŠCHT Praha / ÚOCHB AV ČR</span>
      </div>
    `,
    footerTemplate: `
      <div style="font-size: 7.5pt; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: #64748b; width: 100%; display: flex; justify-content: space-between; align-items: center; padding: 0 12mm; border-top: 0.5pt solid #e2e8f0;">
        <span>Bakalářská Práce: David Kolář · Učební Příručka</span>
        <span style="font-weight: 700; color: #0f172a;">Strana <span class="pageNumber"></span> z <span class="totalPages"></span></span>
      </div>
    `,
    margin: {
      top: '16mm',
      bottom: '16mm',
      left: '12mm',
      right: '12mm'
    }
  });

    await browser.close();

    const stats = fs.statSync(outputPath);
    const sizeMb = (stats.size / (1024 * 1024)).toFixed(2);
    console.log(`✨ Successfully generated: ${outputPath} (${sizeMb} MB)`);
    console.log('=================================================');
  } finally {
    if (serverProc) {
      console.log('🛑 Stopping temporary dev server...');
      serverProc.kill();
    }
  }
}

main().catch(err => {
  console.error('❌ Error generating PDF:', err);
  process.exit(1);
});
