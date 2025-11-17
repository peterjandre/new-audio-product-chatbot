# Frontend - Audio Products Chatbot

TypeScript frontend for the RAG-powered audio products chatbot.

## Local Development

1. Install dependencies:
```bash
npm install
```

2. Build TypeScript:
```bash
npm run build
```

3. Serve the files:
```bash
# Using Python's HTTP server
python -m http.server 3000 -d public

# Or using any static file server
npx serve public
```

4. Open `http://localhost:3000` in your browser.

## Development with Watch Mode

```bash
npm run dev
```

This will watch for TypeScript changes and recompile automatically.

## Vercel Deployment

1. Install Vercel CLI:
```bash
npm i -g vercel
```

2. Set environment variable in Vercel dashboard:
   - `API_BASE_URL` - The URL of your backend API (e.g., `https://your-backend.vercel.app`)

3. Deploy:
```bash
cd frontend
vercel
```

The build process will:
1. Install npm dependencies
2. Compile TypeScript to JavaScript
3. Inject the API_BASE_URL into the HTML

## Project Structure

```
frontend/
├── src/
│   └── script.ts         # TypeScript source
├── public/
│   ├── index.html        # HTML file
│   ├── style.css        # Styles
│   └── script.js        # Compiled JavaScript (generated)
├── package.json         # Node.js dependencies
├── tsconfig.json        # TypeScript configuration
└── vercel.json         # Vercel configuration
```

## Environment Variables

- `API_BASE_URL` - Backend API URL (optional, defaults to relative URLs)

