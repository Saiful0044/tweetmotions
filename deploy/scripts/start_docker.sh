#!/bin/bash
# Login to AWS ECR
aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin 648485681582.dkr.ecr.eu-north-1.amazonaws.com

# Pull the latest image
docker pull 648485681582.dkr.ecr.eu-north-1.amazonaws.com/emotion_ecr:v3

docker stop my-app || true
docker rm my-app || true

# Run a new container
docker run -d -p 80:5000 -e DAGSHUB_PAT=8682bf42823d1ad87472a46fd14c46cda9ab16cf --name my-app 648485681582.dkr.ecr.eu-north-1.amazonaws.com/emotion_ecr:v3