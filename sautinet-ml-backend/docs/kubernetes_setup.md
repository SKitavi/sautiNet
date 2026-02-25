# SentiKenya — Kubernetes Setup Guide

## Overview

Kubernetes orchestrates the distributed SentiKenya services, providing:
- **Scalability**: HPA auto-scales inference pods based on CPU utilisation
- **Fault tolerance**: Automatic pod restart on failure
- **Service discovery**: Internal DNS for inter-service communication

## Architecture

```
                    ┌──────────────────────────────┐
                    │        Kubernetes Cluster     │
                    │                               │
                    │  ┌─────────────────────────┐  │
                    │  │   Inference Deployment   │  │
                    │  │   (3 replicas → HPA)     │  │
                    │  │   ┌─────┐ ┌─────┐ ┌─────┐│  │
                    │  │   │NBO  │ │MSA  │ │KSM  ││  │
                    │  │   └──┬──┘ └──┬──┘ └──┬──┘│  │
                    │  └──────┼───────┼───────┼───┘  │
                    │         │       │       │      │
                    │  ┌──────┴───────┴───────┴───┐  │
                    │  │      kafka-svc:29092      │  │
                    │  └──────────────────────────┘  │
                    │  ┌──────────┐ ┌──────────────┐ │
                    │  │  Redis   │ │  Zookeeper   │ │
                    │  └──────────┘ └──────────────┘ │
                    └──────────────────────────────┘
```

## Prerequisites

Install one of:
- **Minikube**: `brew install minikube` / `choco install minikube`
- **Kind**: `go install sigs.k8s.io/kind@latest`
- **kubectl**: Required for both — `brew install kubectl`

Ensure metrics-server is installed for HPA:
```bash
# Minikube
minikube addons enable metrics-server

# Kind — install manually
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

## Quick Start

### 1. Start Local Cluster

```bash
# Minikube
minikube start --cpus 4 --memory 8192

# Kind
kind create cluster --name sentikenya
```

### 2. Build and Load Docker Image

```bash
# Build
docker build -t sentikenya/inference:latest .

# Load into cluster
minikube image load sentikenya/inference:latest
# or for Kind:
kind load docker-image sentikenya/inference:latest --name sentikenya
```

### 3. Deploy All Services

```bash
# Using kustomize (recommended)
kubectl apply -k kubernetes/

# Or apply individually
kubectl apply -f kubernetes/kafka-deployment.yaml
kubectl apply -f kubernetes/inference-deployment.yaml
kubectl apply -f kubernetes/inference-hpa.yaml
```

### 4. Verify Deployment

```bash
# Check all pods are running
kubectl get pods -n sentikenya

# Expected output:
# NAME                                    READY   STATUS    RESTARTS
# kafka-xxxxx                             1/1     Running   0
# zookeeper-xxxxx                         1/1     Running   0
# redis-xxxxx                             1/1     Running   0
# sentikenya-inference-xxxxx              1/1     Running   0
# sentikenya-inference-yyyyy              1/1     Running   0
# sentikenya-inference-zzzzz              1/1     Running   0

# Check services
kubectl get svc -n sentikenya

# Check HPA
kubectl get hpa -n sentikenya
```

### 5. Access the Inference Service

```bash
# Port-forward to local machine
kubectl port-forward -n sentikenya svc/inference-svc 8000:8000

# Test
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Kenya tech ecosystem is growing fast"}'
```

## HPA Configuration

The Horizontal Pod Autoscaler scales inference pods based on CPU:

| Parameter       | Value  |
|----------------|--------|
| Target CPU     | 70%    |
| Min replicas   | 2      |
| Max replicas   | 6      |
| Scale-up       | +2 pods / 60s stabilisation |
| Scale-down     | -1 pod / 120s stabilisation |

Monitor HPA in real-time:
```bash
kubectl get hpa -n sentikenya -w
```

## Liveness & Readiness Probes

| Probe     | Path              | Initial Delay | Period | Failure Threshold |
|-----------|-------------------|---------------|--------|-------------------|
| Liveness  | `/api/v1/health`  | 30s           | 10s    | 3 (then restart)  |
| Readiness | `/api/v1/health`  | 15s           | 5s     | 5 (no traffic)    |

## Node Failure Simulation

### Test 1: Kill a Pod

```bash
# List inference pods
kubectl get pods -n sentikenya -l component=inference

# Kill one pod
kubectl delete pod <pod-name> -n sentikenya

# Watch it restart automatically
kubectl get pods -n sentikenya -w
# → New pod should reach READY within ~30 seconds
```

### Test 2: Drain a Node (Minikube single-node)

```bash
# Cordon the node (prevent scheduling)
kubectl cordon minikube

# Delete all inference pods (they will be rescheduled when uncordoned)
kubectl delete pods -n sentikenya -l component=inference

# Uncordon
kubectl uncordon minikube

# Verify pods reschedule
kubectl get pods -n sentikenya -w
```

### Test 3: Verify HPA Scaling

```bash
# Generate load (run in separate terminal)
kubectl run -n sentikenya loadtest --image=busybox --restart=Never -- \
  /bin/sh -c "while true; do wget -q -O- http://inference-svc:8000/api/v1/health; done"

# Watch HPA react
kubectl get hpa -n sentikenya -w

# Clean up
kubectl delete pod -n sentikenya loadtest
```

## Teardown

```bash
# Delete all sentikenya resources
kubectl delete -k kubernetes/

# Or delete the whole cluster
minikube delete
# kind delete cluster --name sentikenya
```

## Troubleshooting

| Issue                        | Solution                                              |
|------------------------------|-------------------------------------------------------|
| Pod stuck in `Pending`       | Check resources: `kubectl describe pod <name>`        |
| `ImagePullBackOff`           | Image not loaded into cluster — re-run `minikube image load` |
| HPA shows `<unknown>` CPU    | Metrics-server not installed — enable addon            |
| Pod CrashLoopBackOff         | Check logs: `kubectl logs <pod> -n sentikenya`        |
| Kafka connection refused     | Wait for Kafka readiness probe to pass (~30s)         |
