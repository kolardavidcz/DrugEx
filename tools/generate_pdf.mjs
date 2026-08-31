/**
 * DrugEx Hub — Automated Publication-Grade PDF Generator
 */

import puppeteer from 'puppeteer-core';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

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

async function main() {
  console.log('=================================================');
  console.log(' 📚 DrugEx Hub — Publication PDF Generator ');
  console.log('=================================================');

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
  console.log(`✓ Detected ${slideCount} rendered handbook slide cards across all 6 modules.`);

  console.log('📄 Exporting continuous publication A4 PDF with dark code showcases...');
  await page.pdf({
    path: outputPath,
    format: 'A4',
    printBackground: true,
    displayHeaderFooter: false,
    margin: {
      top: '14mm',
      bottom: '14mm',
      left: '12mm',
      right: '12mm'
    }
  });

  await browser.close();

  const stats = fs.statSync(outputPath);
  const sizeMb = (stats.size / (1024 * 1024)).toFixed(2);
  console.log(`✨ Successfully generated: ${outputPath} (${sizeMb} MB)`);
  console.log('=================================================');
}

main().catch(err => {
  console.error('❌ Error generating PDF:', err);
  process.exit(1);
});
