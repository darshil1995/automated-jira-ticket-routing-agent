#!/bin/bash
# Deploys the latest zip to AWS Lambda

bash scripts/package.sh

echo "Deploying to Lambda..."
aws lambda update-function-code \
  --function-name jira-workflow-agent \
  --zip-file fileb://deployment.zip \
  --region us-east-1

echo "Deployment complete."