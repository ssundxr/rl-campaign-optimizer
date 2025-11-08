# Azure Deployment Script for RL Campaign Optimizer (PowerShell)
# This script deploys the Streamlit dashboard to Azure Container Instances

# Configuration
$RESOURCE_GROUP = "rl-campaign-optimizer-rg"
$LOCATION = "eastus"
$CONTAINER_NAME = "rl-dashboard"
$ACR_NAME = "rlcampaignopt$(Get-Random -Maximum 9999)"  # ACR names must be globally unique
$IMAGE_NAME = "rl-dashboard"
$DNS_NAME = "rl-campaign-opt-$(Get-Random -Maximum 9999)"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Azure Deployment - RL Campaign Optimizer" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Azure CLI
Write-Host "Step 1: Checking Azure CLI..." -ForegroundColor Yellow
try {
    $azVersion = & "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd" --version
    Write-Host "✓ Azure CLI found" -ForegroundColor Green
} catch {
    Write-Host "✗ Azure CLI not found. Please restart PowerShell or add to PATH" -ForegroundColor Red
    exit 1
}

# Step 2: Login to Azure
Write-Host "`nStep 2: Logging into Azure..." -ForegroundColor Yellow
& "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd" login

# Step 3: Create Resource Group
Write-Host "`nStep 3: Creating resource group..." -ForegroundColor Yellow
& "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd" group create `
    --name $RESOURCE_GROUP `
    --location $LOCATION

# Step 4: Create Azure Container Registry
Write-Host "`nStep 4: Creating Azure Container Registry ($ACR_NAME)..." -ForegroundColor Yellow
& "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd" acr create `
    --resource-group $RESOURCE_GROUP `
    --name $ACR_NAME `
    --sku Basic `
    --admin-enabled true

# Step 5: Login to ACR
Write-Host "`nStep 5: Logging into Azure Container Registry..." -ForegroundColor Yellow
& "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd" acr login --name $ACR_NAME

# Step 6: Build and Push Docker Image
Write-Host "`nStep 6: Building and pushing Docker image..." -ForegroundColor Yellow
& "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd" acr build `
    --registry $ACR_NAME `
    --image "${IMAGE_NAME}:latest" `
    --file Dockerfile .

# Step 7: Get ACR credentials
Write-Host "`nStep 7: Getting ACR credentials..." -ForegroundColor Yellow
$ACR_LOGIN_SERVER = (& "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd" acr show `
    --name $ACR_NAME `
    --query loginServer `
    --output tsv)

$ACR_USERNAME = (& "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd" acr credential show `
    --name $ACR_NAME `
    --query username `
    --output tsv)

$ACR_PASSWORD = (& "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd" acr credential show `
    --name $ACR_NAME `
    --query passwords[0].value `
    --output tsv)

# Step 8: Deploy to Azure Container Instances
Write-Host "`nStep 8: Deploying to Azure Container Instances..." -ForegroundColor Yellow
& "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd" container create `
    --resource-group $RESOURCE_GROUP `
    --name $CONTAINER_NAME `
    --image "$ACR_LOGIN_SERVER/${IMAGE_NAME}:latest" `
    --dns-name-label $DNS_NAME `
    --ports 8501 `
    --cpu 2 `
    --memory 4 `
    --registry-login-server $ACR_LOGIN_SERVER `
    --registry-username $ACR_USERNAME `
    --registry-password $ACR_PASSWORD

# Step 9: Get deployment URL
Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "Deployment Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan

$FQDN = (& "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd" container show `
    --resource-group $RESOURCE_GROUP `
    --name $CONTAINER_NAME `
    --query ipAddress.fqdn `
    --output tsv)

Write-Host "`n📊 Dashboard URL: http://${FQDN}:8501" -ForegroundColor Green
Write-Host "`n📋 Useful Commands:" -ForegroundColor Yellow
Write-Host "View logs:" -ForegroundColor Gray
Write-Host "  az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME" -ForegroundColor Gray
Write-Host "`nDelete resources:" -ForegroundColor Gray
Write-Host "  az group delete --name $RESOURCE_GROUP --yes" -ForegroundColor Gray
Write-Host ""
