#!/bin/bash
# Build script for Vercel deployment

# Install dependencies
npm install

# Compile TypeScript
npm run build

# Replace API_BASE_URL placeholder in HTML if environment variable is set
if [ -n "$API_BASE_URL" ]; then
    sed -i.bak "s|{{API_BASE_URL}}|$API_BASE_URL|g" public/index.html
    rm public/index.html.bak
else
    sed -i.bak "s|{{API_BASE_URL}}||g" public/index.html
    rm public/index.html.bak
fi

