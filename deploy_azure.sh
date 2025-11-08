#!/bin/bash

# Azure Deployment Script for RL Campaign Optimizer
# This script deploys the Streamlit dashboard to Azure Container Instances

set -e

# Configuration
RESOURCE_GROUP="rl-campaign-optimizer-rg"
LOCATION="eastus"
CONTAINER_NAME="rl-dashboard"
ACR_NAME="rlcampaignoptimizer"
IMAGE_NAME="rl-dashboard"
DNS_NAME="rl-campaign-optimizer"

echo "=========================================="
echo "Azure Deployment - RL Campaign Optimizer"
echo "=========================================="

# Step 1: Login to Azure
echo "Step 1: Logging into Azure..."
az login

# Step 2: Create Resource Group
echo "Step 2: Creating resource group..."
az group create --name $RESOURCE_GROUP --location $LOCATION

# Step 3: Create Azure Container Registry
echo "Step 3: Creating Azure Container Registry..."
az acr create \
    --resource-group $RESOURCE_GROUP \
    --name $ACR_NAME \
    --sku Basic \
    --admin-enabled true

# Step 4: Login to ACR
echo "Step 4: Logging into Azure Container Registry..."
az acr login --name $ACR_NAME

# Step 5: Build and Push Docker Image
echo "Step 5: Building and pushing Docker image..."
az acr build \
    --registry $ACR_NAME \
    --image $IMAGE_NAME:latest \
    --file Dockerfile .

# Step 6: Get ACR credentials
echo "Step 6: Getting ACR credentials..."
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --query loginServer --output tsv)
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username --output tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query passwords[0].value --output tsv)

# Step 7: Deploy to Azure Container Instances
echo "Step 7: Deploying to Azure Container Instances..."
az container create \
    --resource-group $RESOURCE_GROUP \
    --name $CONTAINER_NAME \
    --image $ACR_LOGIN_SERVER/$IMAGE_NAME:latest \
    --dns-name-label $DNS_NAME \
    --ports 8501 \
    --cpu 2 \
    --memory 4 \
    --registry-login-server $ACR_LOGIN_SERVER \
    --registry-username $ACR_USERNAME \
    --registry-password $ACR_PASSWORD

# Step 8: Get deployment URL
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
FQDN=$(az container show \
    --resource-group $RESOURCE_GROUP \
    --name $CONTAINER_NAME \
    --query ipAddress.fqdn \
    --output tsv)

echo "Dashboard URL: http://$FQDN:8501"
echo ""
echo "To view logs:"
echo "az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME"
echo ""
echo "To delete resources:"
echo "az group delete --name $RESOURCE_GROUP --yes"
