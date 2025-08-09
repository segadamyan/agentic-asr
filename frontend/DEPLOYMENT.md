# Frontend Deployment to GitHub Pages

This React frontend is automatically deployed to GitHub Pages using GitHub Actions.

## 🚀 Live Demo

The application is deployed at: **https://segadamyan.github.io/agentic-asr**

## 📦 Deployment Setup

### Automatic Deployment (Recommended)

The application is automatically deployed when changes are pushed to the `main` branch using GitHub Actions.

### Manual Deployment

To deploy manually:

```bash
cd frontend
npm install
npm run deploy
```

## 🔧 Configuration

The deployment uses the following configuration:

- **Homepage URL**: `https://segadamyan.github.io/agentic-asr`
- **Base Path**: `/agentic-asr` (configured in React Router)
- **GitHub Pages Branch**: `gh-pages`

## 📁 Key Files for Deployment

- `.github/workflows/deploy.yml` - GitHub Actions workflow
- `frontend/package.json` - Contains homepage URL and deploy scripts
- `frontend/public/404.html` - Handles client-side routing for SPA
- `frontend/public/index.html` - Contains SPA routing script
- `frontend/src/App.tsx` - Router configured with basename

## 🛠 Troubleshooting

### Routing Issues
- The app uses React Router with a basename of `/agentic-asr`
- SPA routing is handled by `404.html` and scripts in `index.html`

### Build Issues
- Make sure all dependencies are installed: `npm install`
- Check for TypeScript/ESLint errors before deployment
- The build assumes hosting at `/agentic-asr/` subdirectory

### GitHub Pages Settings
1. Go to repository Settings → Pages
2. Source should be set to "Deploy from a branch"
3. Branch should be `gh-pages` / `/ (root)`

## 🔄 Development vs Production

### Development
```bash
npm start  # Runs on localhost:3000 with proxy to backend
```

### Production
- Built assets are served from GitHub Pages
- API calls need to be configured for production backend URL
- No proxy configuration in production build

## 📋 Next Steps

After deployment, you may need to:

1. **Configure API endpoints** for production
2. **Set up CORS** on your backend for the GitHub Pages domain
3. **Update any hardcoded localhost URLs** in the frontend code
4. **Configure environment variables** for different environments

---

*The frontend will be available at the GitHub Pages URL once the GitHub Actions workflow completes successfully.*
