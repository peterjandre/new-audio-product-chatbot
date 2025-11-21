/**
 * Build script to inject environment variables into HTML
 * This runs after TypeScript compilation
 */
const fs = require('fs');
const path = require('path');

// Try multiple possible paths for index.html (Vercel build environment may have different working directory)
const possiblePaths = [
    path.join(__dirname, 'public', 'index.html'),
    path.join(process.cwd(), 'public', 'index.html'),
    path.join(process.cwd(), 'frontend', 'public', 'index.html'),
];

let htmlPath = null;
for (const possiblePath of possiblePaths) {
    if (fs.existsSync(possiblePath)) {
        htmlPath = possiblePath;
        break;
    }
}

if (!htmlPath) {
    console.error('index.html not found. Tried paths:');
    possiblePaths.forEach(p => console.error('  -', p));
    console.error('Current working directory:', process.cwd());
    console.error('__dirname:', __dirname);
    process.exit(1);
}

// Use API_BASE_URL if set, otherwise leave empty for relative URLs
const apiBaseUrl = process.env.API_BASE_URL || '';

try {
    let html = fs.readFileSync(htmlPath, 'utf8');
    html = html.replace('{{API_BASE_URL}}', apiBaseUrl);
    fs.writeFileSync(htmlPath, html, 'utf8');
    console.log(`✓ Injected API_BASE_URL into ${htmlPath}`);
    console.log(`  API_BASE_URL: ${apiBaseUrl || '(empty - will use relative URLs)'}`);
} catch (error) {
    console.error('Error processing index.html:', error);
    process.exit(1);
}

