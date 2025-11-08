# Quick Azure Deployment Script
# Run this after installing Azure CLI: winget install Microsoft.AzureCLI

# Configuration
$resourceGroup = "rl-campaign-optimizer-rg"
$location = "eastus"
$appServicePlan = "rl-optimizer-plan"
$webAppName = "rl-campaign-optimizer-dashboard"
$runtime = "PYTHON:3.11"
$sku = "B1"

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "Azure Deployment for RL Campaign Optimizer" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

# Check if Azure CLI is installed
try {
    az --version | Out-Null
} catch {
    Write-Host "ERROR: Azure CLI not found!" -ForegroundColor Red
    Write-Host "Install with: winget install Microsoft.AzureCLI" -ForegroundColor Yellow
    exit 1
}

Write-Host "Step 1: Logging into Azure..." -ForegroundColor Green
az login

Write-Host "`nStep 2: Creating Resource Group..." -ForegroundColor Green
az group create --name $resourceGroup --location $location

Write-Host "`nStep 3: Creating App Service Plan..." -ForegroundColor Green
az appservice plan create `
    --name $appServicePlan `
    --resource-group $resourceGroup `
    --sku $sku `
    --is-linux

Write-Host "`nStep 4: Creating Web App..." -ForegroundColor Green
az webapp create `
    --resource-group $resourceGroup `
    --plan $appServicePlan `
    --name $webAppName `
    --runtime $runtime

Write-Host "`nStep 5: Configuring App Settings..." -ForegroundColor Green
az webapp config appsettings set `
    --resource-group $resourceGroup `
    --name $webAppName `
    --settings SCM_DO_BUILD_DURING_DEPLOYMENT=true WEBSITE_PORT=8501

Write-Host "`nStep 6: Setting Startup Command..." -ForegroundColor Green
az webapp config set `
    --resource-group $resourceGroup `
    --name $webAppName `
    --startup-file "startup.sh"

Write-Host "`nStep 7: Deploying Application..." -ForegroundColor Green
az webapp up `
    --resource-group $resourceGroup `
    --name $webAppName `
    --runtime $runtime `
    --sku $sku

Write-Host "`n==================================" -ForegroundColor Cyan
Write-Host "DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Dashboard URL: https://$webAppName.azurewebsites.net" -ForegroundColor Yellow
Write-Host ""
Write-Host "To view logs: az webapp log tail --resource-group $resourceGroup --name $webAppName" -ForegroundColor Gray
Write-Host "To delete all resources: az group delete --name $resourceGroup --yes" -ForegroundColor Gray
Write-Host ""
