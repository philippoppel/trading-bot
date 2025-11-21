# Cloud Deployment Guide

Complete guide to deploy your trading bot to the cloud so everything runs independently of your PC.

## Architecture

```
Railway.app (Trading Bot)
    ↓ uploads state every 30s
Vercel Blob (Storage)
    ↑ reads state
Vercel (Dashboard)
```

Everything runs in the cloud - no PC needed!

## Prerequisites

1. GitHub account
2. Railway.app account (free tier available)
3. Vercel account (already set up)
4. Vercel Blob storage (see VERCEL_SETUP.md)

## Step 1: Set Up Vercel Blob Storage

Follow the instructions in `trading-dashboard/VERCEL_SETUP.md`:

1. Create Blob storage in Vercel dashboard
2. Set `BLOB_READ_WRITE_TOKEN` (auto-generated)
3. Generate and set `UPLOAD_API_KEY`:
   ```bash
   openssl rand -hex 32
   ```

## Step 2: Create GitHub Repository for Trading Bot

```bash
cd /Users/philippoppel/Desktop/traidingbot

# Initialize git (if not already done)
git init
git add .
git commit -m "Initial commit: Cloud trading bot"

# Create GitHub repo
gh repo create trading-bot --public --source=. --remote=origin --push
```

## Step 3: Deploy to Railway.app

### 3.1 Sign Up / Log In

1. Go to https://railway.app
2. Sign up with GitHub

### 3.2 Create New Project

1. Click "New Project"
2. Select "Deploy from GitHub repo"
3. Choose your `trading-bot` repository
4. Railway will auto-detect the Dockerfile

### 3.3 Set Environment Variables

In Railway dashboard, go to **Variables** and add:

```
VERCEL_DASHBOARD_URL=https://trading-dashboard-5oqf34l8u-philipps-projects-0f51423d.vercel.app
UPLOAD_API_KEY=<your-generated-api-key>
TRADING_INTERVAL=3600
UPLOAD_INTERVAL=30
```

**Important:** Use the SAME `UPLOAD_API_KEY` that you set in Vercel!

### 3.4 Deploy

1. Click "Deploy"
2. Railway will build the Docker image and start your bot
3. Check the logs to verify it's running

## Step 4: Verify Everything Works

### 4.1 Check Railway Logs

In Railway dashboard:
- Go to your deployment
- Click **"Deployments"** → **"View Logs"**
- You should see trading activity

### 4.2 Check Dashboard

1. Go to your dashboard: https://trading-dashboard-5oqf34l8u-philipps-projects-0f51423d.vercel.app
2. Within 30 seconds you should see live trading data
3. The dashboard auto-refreshes every 10 seconds

## Configuration

### Trading Interval

Default: 1 hour (3600 seconds)

To change, set `TRADING_INTERVAL` in Railway:
```
TRADING_INTERVAL=1800  # 30 minutes
```

### Upload Interval

Default: 30 seconds

To change, set `UPLOAD_INTERVAL` in Railway:
```
UPLOAD_INTERVAL=60  # 1 minute
```

## Costs

### Railway.app

- **Free Tier:** $5 credit/month
- **Hobby Plan:** $5/month flat rate (unlimited usage)
- Your bot should run fine on the free tier for testing

### Vercel

- **Hobby (Free):** Perfect for this dashboard
- Blob storage: First 1GB free, then $0.15/GB

## Monitoring

### Railway Dashboard

- View real-time logs
- Check CPU/Memory usage
- Restart if needed

### Vercel Dashboard

- View deployment status
- Check API request metrics
- Monitor Blob storage usage

## Troubleshooting

### Bot not uploading to Vercel

1. Check Railway logs for upload errors
2. Verify `UPLOAD_API_KEY` matches in both Railway and Vercel
3. Verify `VERCEL_DASHBOARD_URL` is correct

### Dashboard shows "State not found"

1. Wait 30 seconds after bot starts
2. Check Railway logs to see if uploads are successful
3. Verify Blob storage is created in Vercel

### Bot crashes on startup

1. Check Railway logs for error messages
2. Verify all environment variables are set
3. Check that models directory exists and contains trained models

## Stopping the Bot

### Temporary Stop

In Railway dashboard:
1. Go to your service
2. Click **"Settings"** → **"Sleep"**

### Permanent Stop

In Railway dashboard:
1. Go to your service
2. Click **"Settings"** → **"Delete Service"**

## Updating the Bot

When you push changes to GitHub:
1. Railway automatically detects the change
2. Builds a new Docker image
3. Deploys the updated version
4. Zero downtime deployment

```bash
# Make changes to code
git add .
git commit -m "Update trading strategy"
git push

# Railway automatically deploys!
```

## Support

- Railway Docs: https://docs.railway.app
- Vercel Docs: https://vercel.com/docs
- GitHub Issues: Create an issue in your repository

## Next Steps

Once everything is running:

1. Monitor performance for 24 hours
2. Check dashboard regularly
3. Review trading logs
4. Adjust parameters as needed

Your trading bot is now fully cloud-native! 🚀
