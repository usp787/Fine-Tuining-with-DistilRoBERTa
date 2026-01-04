# LoRA Emotion Classifier - Deployment Guide

Complete guide to deploy your DistilRoBERTa emotion classifier to AWS.

## 📋 Prerequisites

- ✅ Docker installed
- ✅ AWS CLI installed and configured
- ✅ Your trained model file: `best_LoRA_model.pt`
- ✅ AWS account with appropriate permissions

## 🧪 Step 1: Test Locally First

### 1.1 Build the Docker image

```bash
docker build -t lora-emotion-api:latest .
```

### 1.2 Run locally (CPU mode for testing)

```bash
docker run -p 8080:8080 \
  -e DEVICE=cpu \
  lora-emotion-api:latest
```

Or with GPU (if you have NVIDIA GPU):

```bash
docker run --gpus all -p 8080:8080 \
  -e DEVICE=cuda \
  lora-emotion-api:latest
```

### 1.3 Test the API

Open another terminal and run:

```bash
# Install requests if needed
pip install requests

# Run test script
python test_api.py
```

Or test manually with curl:

```bash
# Health check
curl http://localhost:8080/health

# Single prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I am so happy today!", "top_k": 3}'

# Batch prediction
curl -X POST http://localhost:8080/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "I love this!",
      "This is terrible.",
      "Not sure what to think."
    ],
    "threshold": 0.5
  }'
```

## 🚀 Step 2: Deploy to AWS

### Option A: AWS Lambda (Cheapest for Low Traffic)

⚠️ **Limitation**: Lambda has a 10GB container size limit. Your model + dependencies might exceed this.

**Alternative**: Use Lambda with model stored in S3 (modify code to download model at startup).

Skip to **Option B** or **Option C** for simpler deployment.

---

### Option B: AWS EC2 (Recommended for Learning)

Best for: Testing, learning, and consistent traffic. **Free tier eligible!**

#### B.1 Push image to ECR

```bash
# Set your AWS region and account ID
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Create ECR repository
aws ecr create-repository --repository-name lora-emotion-api --region $AWS_REGION

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Tag your image
docker tag lora-emotion-api:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/lora-emotion-api:latest

# Push to ECR
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/lora-emotion-api:latest
```

#### B.2 Launch EC2 Instance

**Option 1: Using AWS Console (Easiest)**

1. Go to EC2 Console → Launch Instance
2. **Name**: `lora-emotion-api`
3. **AMI**: Amazon Linux 2023 (free tier eligible)
4. **Instance type**: 
   - For CPU: `t3.medium` or `t3.large` (NOT t3.micro - too small)
   - For GPU: `g4dn.xlarge` (NOT free tier, ~$0.50/hour)
5. **Key pair**: Create or select existing
6. **Security Group**: 
   - Allow SSH (port 22) from your IP
   - Allow HTTP (port 80) from anywhere
   - Allow Custom TCP (port 8080) from anywhere
7. **Storage**: 30 GB (minimum for model + dependencies)
8. Launch!

**Option 2: Using AWS CLI**

```bash
# Create security group
aws ec2 create-security-group \
  --group-name lora-api-sg \
  --description "Security group for LoRA API"

# Get the security group ID
SG_ID=$(aws ec2 describe-security-groups \
  --group-names lora-api-sg \
  --query 'SecurityGroups[0].GroupId' \
  --output text)

# Allow SSH, HTTP, and API access
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp --port 22 --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp --port 80 --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp --port 8080 --cidr 0.0.0.0/0

# Launch instance (t3.medium for CPU)
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type t3.medium \
  --key-name YOUR_KEY_PAIR_NAME \
  --security-group-ids $SG_ID \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":30}}]'
```

#### B.3 Connect and Deploy

```bash
# SSH into your instance
ssh -i lora_emotion_api_key.pem ec2-user@54.153.54.135 # public IP: 54.153.54.135; key: lora_emotion_api_key.pem

# Install Docker
sudo yum update -y
sudo yum install docker -y
sudo service docker start
sudo usermod -a -G docker ec2-user

# Logout and login again for group changes to take effect
exit
ssh -i lora_emotion_api_key.pem ec2-user@54.153.54.135

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Pull and run your container
docker pull $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/lora-emotion-api:latest

docker run -d --name emotion-api --restart unless-stopped -p 80:8080 -e DEVICE=cpu $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/lora-emotion-api:latest

# Check logs
docker logs -f emotion-api
```

#### B.4 Test Your Deployment

```bash
# From your local machine
curl http://54.153.54.135/health

curl -X POST http://54.153.54.135/predict -H "Content-Type: application/json" -d '{"text": "I am so excited!", "top_k": 3}'
```

---

### Option C: AWS Fargate (Best for Production)

Best for: Automatic scaling, no server management, pay only when running.

#### C.1 Create ECS Cluster

```bash
aws ecs create-cluster --cluster-name lora-emotion-cluster
```

#### C.2 Create Task Definition

Save this as `task-definition.json`:

```json
{
  "family": "lora-emotion-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "lora-emotion-api",
      "image": "<YOUR_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/lora-emotion-api:latest",
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "DEVICE", "value": "cpu"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/lora-emotion-api",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

Register it:

```bash
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

#### C.3 Create Service with Load Balancer

This is complex - I recommend using the AWS Console:

1. Go to ECS → Clusters → Create Service
2. Choose Fargate, select your task definition
3. Create Application Load Balancer
4. Configure target group (port 8080)
5. Set health check path: `/health`
6. Launch!

---

## 💰 Cost Comparison

### EC2 t3.medium (CPU)
- **Free Tier**: 750 hours/month for first 12 months
- **After**: ~$30/month (if running 24/7)
- **Best for**: Testing, learning, consistent low traffic

### EC2 g4dn.xlarge (GPU)
- **Cost**: ~$0.50/hour = ~$360/month (24/7)
- **Best for**: High-performance needs, batch processing

### Fargate
- **Cost**: ~$30-50/month for 1 vCPU, 2GB RAM running 24/7
- **Best for**: Production, automatic scaling

### Lambda
- **Cost**: First 1M requests free, then $0.20/1M
- **Best for**: Sporadic usage, very low traffic
- **Note**: May need modifications for model storage

---

## 🔧 Monitoring & Maintenance

### View Logs (EC2)

```bash
ssh -i your-key.pem ec2-user@<EC2_PUBLIC_IP>
docker logs -f emotion-api
```

### Update Model (EC2)

```bash
# Build new image locally
docker build -t lora-emotion-api:latest .

# Push to ECR
docker tag lora-emotion-api:latest \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/lora-emotion-api:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/lora-emotion-api:latest

# SSH to EC2 and update
ssh -i your-key.pem ec2-user@<EC2_PUBLIC_IP>
docker stop emotion-api
docker rm emotion-api
docker pull $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/lora-emotion-api:latest
docker run -d --name emotion-api --restart unless-stopped -p 80:8080 \
  -e DEVICE=cpu \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/lora-emotion-api:latest
```

### Monitor Performance

```bash
# Check container stats
docker stats emotion-api

# Check API metrics
curl http://<EC2_PUBLIC_IP>/health
```

---

## 🛡️ Security Best Practices

1. **Don't use root AWS account** - Create IAM user
2. **Restrict security groups** - Only allow necessary ports
3. **Use HTTPS** - Set up SSL certificate (use AWS Certificate Manager + ALB)
4. **API authentication** - Add API keys if needed (modify `app.py`)
5. **Rate limiting** - Consider adding rate limits to prevent abuse

---

## 🎯 Quick Start Summary

**For absolute beginners, I recommend:**

1. ✅ Test locally with Docker first
2. ✅ Deploy to EC2 t3.medium (free tier)
3. ✅ Once comfortable, consider Fargate for production

**Simplest deployment path:**
```bash
# 1. Build
docker build -t lora-emotion-api .

# 2. Push to ECR
aws ecr create-repository --repository-name lora-emotion-api
# ... login and push commands from above ...

# 3. Launch EC2 via console (t3.medium)

# 4. SSH and run
docker run -d -p 80:8080 --restart unless-stopped \
  <your-ecr-image>
```

---

## 📚 Additional Resources

- [AWS Free Tier](https://aws.amazon.com/free/)
- [EC2 Pricing](https://aws.amazon.com/ec2/pricing/)
- [Fargate Pricing](https://aws.amazon.com/fargate/pricing/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

## ❓ Troubleshooting

**Container won't start?**
- Check logs: `docker logs emotion-api`
- Verify model file exists in container
- Check memory limits

**Can't connect to API?**
- Verify security group allows port 8080 or 80
- Check if container is running: `docker ps`
- Test locally first: `curl http://localhost:8080/health`

**Out of memory?**
- Use larger instance type (t3.large or t3.xlarge)
- Reduce batch size in requests
- Consider using CPU instead of loading model on GPU

**Need help?**
- Check CloudWatch Logs (AWS Console)
- Review container logs
- Test locally to isolate AWS vs. code issues
