/**
 * Build script to inject environment variables into HTML
 * This runs after TypeScript compilation
 */
const fs = require('fs');
const path = require('path');

const htmlPath = path.join(__dirname, 'public', 'index.html');
// Use API_BASE_URL if set, otherwise leave empty for relative URLs
const apiBaseUrl = process.env.API_BASE_URL || '';

if (fs.existsSync(htmlPath)) {
    let html = fs.readFileSync(htmlPath, 'utf8');
    html = html.replace('{{API_BASE_URL}}', apiBaseUrl);
    fs.writeFileSync(htmlPath, html, 'utf8');
    console.log(`Injected API_BASE_URL: ${apiBaseUrl || '(empty - will use relative URLs)'}`);
} else {
    console.error('index.html not found at', htmlPath);
    process.exit(1);
}

