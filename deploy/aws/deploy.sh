#!/usr/bin/env bash
# Build + push to ECR and run on AWS App Runner. Requires: aws configure, docker.
set -euo pipefail
REGION="${AWS_REGION:-ap-south-1}"; ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REPO="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/aeroops-api"; TAG=$(git rev-parse --short HEAD)
aws ecr describe-repositories --repository-names aeroops-api --region "$REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name aeroops-api --region "$REGION"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "${REPO%/*}"
docker build -t "$REPO:$TAG" . && docker push "$REPO:$TAG"
sed "s#IMAGE_URI#$REPO:$TAG#; s#ACCOUNT_ID#$ACCOUNT#" deploy/aws/apprunner.json > /tmp/apprunner.json
aws apprunner create-service --cli-input-json file:///tmp/apprunner.json --region "$REGION"
