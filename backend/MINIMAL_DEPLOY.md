# Minimal API Deployment Guide

This is a barebones FastAPI app to test if Vercel deployment works at all.

## Files Created

1. `api/index_minimal.py` - Minimal FastAPI app with just 2 endpoints
2. `requirements_minimal.txt` - Only FastAPI and Mangum dependencies
3. `vercel_minimal.json` - Vercel config pointing to minimal handler

## To Deploy Minimal Version

### Option 1: Temporarily Replace Files

1. **Backup current files:**
   ```bash
   cd backend
   cp api/index.py api/index.py.backup
   cp vercel.json vercel.json.backup
   cp requirements.txt requirements.txt.backup
   ```

2. **Replace with minimal versions:**
   ```bash
   cp api/index_minimal.py api/index.py
   cp vercel_minimal.json vercel.json
   cp requirements_minimal.txt requirements.txt
   ```

3. **Deploy:**
   ```bash
   vercel --prod
   ```

4. **Test:** Visit your Vercel URL - you should see `{"message": "Hello from Vercel!", "status": "ok"}`

5. **Restore original files if needed:**
   ```bash
   cp api/index.py.backup api/index.py
   cp vercel.json.backup vercel.json
   cp requirements.txt.backup requirements.txt
   ```

### Option 2: Create Separate Vercel Project

1. Create a new directory for minimal test:
   ```bash
   mkdir backend-minimal
   cd backend-minimal
   ```

2. Copy minimal files:
   ```bash
   cp ../backend/api/index_minimal.py api/index.py
   cp ../backend/vercel_minimal.json vercel.json
   cp ../backend/requirements_minimal.txt requirements.txt
   cp ../backend/runtime.txt runtime.txt
   ```

3. Deploy as new project:
   ```bash
   vercel
   ```

## What This Tests

- ✅ Vercel can deploy Python functions
- ✅ FastAPI works on Vercel
- ✅ Mangum adapter works
- ✅ Basic routing works

If this works, then the issue is with:
- Complex imports (app.py, scripts)
- Large dependencies (faiss-cpu, numpy)
- File downloads
- Environment variables

## Next Steps

Once minimal version works:
1. Add back imports one at a time
2. Add dependencies one at a time
3. Test each addition

