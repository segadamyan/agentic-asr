# 🚀 Railway Deployment Guide

## 🌟 **Deploy to Railway**

### Quick Deploy Steps
1. Go to [Railway.app](https://railway.app)
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select `segadamyan/agentic-asr`
5. Railway will auto-deploy your backend!

### Environment Variables
After deployment, add these in Railway dashboard:
```
OPENAI_API_KEY=your_actual_openai_key
DEFAULT_LLM_PROVIDER=openai
DEFAULT_LLM_MODEL=gpt-4o
```

### ⚡ **Optimized Build**
The deployment uses:
- **Nixpacks** for faster Python builds
- **Lightweight requirements** (`railway-requirements.txt`)
- **Optional audio processing** (transcription features disabled for faster deployment)

### 📋 **Available Features**
✅ **Chat with AI Agent**  
✅ **Text Analysis & Correction**  
✅ **Summarization & Translation**  
✅ **Conversation History**  
✅ **Session Management**  
❌ **Audio File Upload** (requires additional setup)

### Your Backend URL
Railway will provide a URL like: `https://agentic-asr-production-XXXX.up.railway.app`

---

## � **After Backend Deployment**

### 1. Copy Your Railway URL
After deployment, copy the Railway-provided URL from your dashboard.

### 2. Update Frontend API URL
Edit `frontend/.env.production` and replace the placeholder with your actual Railway URL:
```bash
REACT_APP_API_URL=https://your-actual-railway-url.railway.app
```

### 3. Redeploy Frontend
```bash
cd frontend
npm run deploy
```

### 4. Test the Connection
Visit your GitHub Pages site and test the full-stack integration!

---

## � **Troubleshooting**

### Build Issues
- Railway automatically detects Python and installs dependencies
- If build fails, check the Railway logs for specific errors

### Environment Variables
- Make sure `OPENAI_API_KEY` is set in Railway dashboard
- Verify the API key has sufficient credits

### CORS Issues
- The backend is already configured to accept requests from GitHub Pages
- If you get CORS errors, double-check the frontend URL configuration

---

## 🎯 **Complete Workflow**

1. ✅ **Deploy Backend**: Railway auto-deploys from GitHub
2. ✅ **Configure Environment**: Add `OPENAI_API_KEY` in Railway
3. ⏳ **Update Frontend**: Edit `.env.production` with Railway URL
4. ⏳ **Redeploy Frontend**: Run `npm run deploy`
5. ⏳ **Test**: Visit GitHub Pages site

Your full-stack Agentic ASR app will be live! 🎉
