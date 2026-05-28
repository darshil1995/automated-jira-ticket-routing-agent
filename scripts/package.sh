#!/bin/bash
# Packages the Lambda deployment zip

echo "Cleaning old package..."
rm -rf package deployment.zip

echo "Installing dependencies..."
pip install --target ./package -r requirements.txt

echo "Copying application code..."
cp -r agents utils graph.py lambda_handler.py config.py ./package/

echo "Zipping..."
cd package && zip -r ../deployment.zip . && cd ..

echo "Done. deployment.zip ready."
ls -lh deployment.zip