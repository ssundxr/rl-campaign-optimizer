# Azure Deployment Guide for RL Campaign Optimizer

## Prerequisites
1. Azure CLI installed (run: `winget install Microsoft.AzureCLI`)
2. Azure subscription active
3. Docker Desktop running (for container deployment option)

## Deployment Options

### Option 1: Azure App Service (Recommended for Streamlit)

#### Step 1: Login to Azure
```bash
az login
```

#### Step 2: Create Resource Group
```bash
az group create --name rl-campaign-optimizer-rg --location eastus
```

#### Step 3: Create App Service Plan (Linux)
```bash
az appservice plan create \
  --name rl-optimizer-plan \
  --resource-group rl-campaign-optimizer-rg \
  --sku B1 \
  --is-linux
```

#### Step 4: Create Web App
```bash
az webapp create \
  --resource-group rl-campaign-optimizer-rg \
  --plan rl-optimizer-plan \
  --name rl-campaign-optimizer-dashboard \
  --runtime "PYTHON:3.11"
```

#### Step 5: Configure App Settings
```bash
az webapp config appsettings set \
  --resource-group rl-campaign-optimizer-rg \
  --name rl-campaign-optimizer-dashboard \
  --settings SCM_DO_BUILD_DURING_DEPLOYMENT=true \
             WEBSITE_PORT=8501
```

#### Step 6: Configure Startup Command
```bash
az webapp config set \
  --resource-group rl-campaign-optimizer-rg \
  --name rl-campaign-optimizer-dashboard \
  --startup-file "startup.sh"
```

#### Step 7: Deploy Code
```bash
# From project root directory
az webapp up \
  --resource-group rl-campaign-optimizer-rg \
  --name rl-campaign-optimizer-dashboard \
  --runtime "PYTHON:3.11" \
  --sku B1
```

#### Step 8: Access Dashboard
```
https://rl-campaign-optimizer-dashboard.azurewebsites.net
```

---

### Option 2: Azure Container Instances (Docker)

#### Step 1: Create Dockerfile (already exists in project)

#### Step 2: Build and Push to Azure Container Registry
```bash
# Create ACR
az acr create \
  --resource-group rl-campaign-optimizer-rg \
  --name rloptimizeracr \
  --sku Basic

# Login to ACR
az acr login --name rloptimizeracr

# Build and push
az acr build \
  --registry rloptimizeracr \
  --image rl-dashboard:v1 \
  --file Dockerfile.azure .
```

#### Step 3: Deploy Container
```bash
az container create \
  --resource-group rl-campaign-optimizer-rg \
  --name rl-dashboard-container \
  --image rloptimizeracr.azurecr.io/rl-dashboard:v1 \
  --dns-name-label rl-campaign-optimizer \
  --ports 8501 \
  --cpu 2 \
  --memory 4
```

---

### Option 3: Azure Static Web Apps + Functions (API)

For production with separate backend:

1. **Deploy Streamlit to App Service** (Option 1)
2. **Deploy Flask API to Azure Functions**
3. **Use Azure PostgreSQL** for database
4. **Use Azure Event Hubs** instead of Kafka

---

## Cost Estimation

### Basic Setup (B1 App Service):
- **App Service Plan (B1)**: ~$13/month
- **Storage**: ~$1/month
- **Total**: ~$14/month

### Production Setup:
- **App Service Plan (P1V2)**: ~$73/month
- **Azure PostgreSQL**: ~$40/month
- **Azure Event Hubs**: ~$11/month
- **Storage + Network**: ~$10/month
- **Total**: ~$134/month

---

## Post-Deployment Configuration

### 1. Enable HTTPS (Automatic with App Service)
```bash
az webapp update \
  --resource-group rl-campaign-optimizer-rg \
  --name rl-campaign-optimizer-dashboard \
  --https-only true
```

### 2. Scale Up/Out
```bash
# Scale up (more CPU/RAM)
az appservice plan update \
  --name rl-optimizer-plan \
  --resource-group rl-campaign-optimizer-rg \
  --sku P1V2

# Scale out (more instances)
az appservice plan update \
  --name rl-optimizer-plan \
  --resource-group rl-campaign-optimizer-rg \
  --number-of-workers 3
```

### 3. Configure Custom Domain
```bash
az webapp config hostname add \
  --resource-group rl-campaign-optimizer-rg \
  --webapp-name rl-campaign-optimizer-dashboard \
  --hostname yourdomain.com
```

### 4. Enable Application Insights (Monitoring)
```bash
az monitor app-insights component create \
  --app rl-optimizer-insights \
  --location eastus \
  --resource-group rl-campaign-optimizer-rg \
  --application-type web

# Link to Web App
az webapp config appsettings set \
  --resource-group rl-campaign-optimizer-rg \
  --name rl-campaign-optimizer-dashboard \
  --settings APPINSIGHTS_INSTRUMENTATIONKEY=<your-key>
```

---

## Monitoring & Logs

### View Logs
```bash
az webapp log tail \
  --resource-group rl-campaign-optimizer-rg \
  --name rl-campaign-optimizer-dashboard
```

### Stream Logs
```bash
az webapp log download \
  --resource-group rl-campaign-optimizer-rg \
  --name rl-campaign-optimizer-dashboard \
  --log-file logs.zip
```

---

## Troubleshooting

### Issue: App not starting
**Solution**: Check startup.sh permissions and Python version

### Issue: Model file not found
**Solution**: Ensure models/ directory is included in deployment (not in .gitignore for trained models)

### Issue: Port binding error
**Solution**: Verify WEBSITE_PORT=8501 is set in App Settings

### Issue: Dependencies not installing
**Solution**: Check requirements-azure.txt exists and Python version matches

---

## Cleanup (Delete Resources)

```bash
az group delete \
  --name rl-campaign-optimizer-rg \
  --yes \
  --no-wait
```

---

## GitHub Actions CI/CD (Optional)

Create `.github/workflows/azure-deploy.yml` for automatic deployment on push to main branch.

---

## Security Best Practices

1. **Enable Azure AD Authentication**
2. **Use Managed Identity** for database connections
3. **Store secrets in Azure Key Vault**
4. **Enable DDoS Protection**
5. **Configure CORS** properly
6. **Use Private Endpoints** for database

---

## Support

- Azure Documentation: https://learn.microsoft.com/en-us/azure/
- Streamlit Cloud: https://streamlit.io/cloud (alternative)
- GitHub Issues: https://github.com/ssundxr/rl-campaign-optimizer/issues
