#!/usr/bin/env bash
# CreditPulse — minikube K8s deployment
# One-shot script: starts minikube, builds images, deploys all services
# "Architected for AWS EKS; local demo runs on minikube."

set -euo pipefail

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " CreditPulse — minikube Deploy"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 1. Start minikube ──────────────────────────────────────
if ! minikube status | grep -q "Running"; then
    echo "[1/7] Starting minikube..."
    minikube start --cpus=4 --memory=8192 --disk-size=30g --driver=docker
else
    echo "[1/7] minikube already running ✓"
fi

# ── 2. Use minikube Docker daemon ─────────────────────────
eval "$(minikube docker-env)"

# ── 3. Build Docker images ─────────────────────────────────
echo "[2/7] Building API image..."
docker build -t creditpulse-api:latest -f infra/docker/Dockerfile.api .

echo "[3/7] Building frontend image..."
docker build -t creditpulse-frontend:latest -f infra/docker/Dockerfile.frontend frontend/

# ── 4. Install KEDA ───────────────────────────────────────
echo "[4/7] Installing KEDA..."
if ! kubectl get ns keda &>/dev/null; then
    helm repo add kedacore https://kedacore.github.io/charts 2>/dev/null || true
    helm repo update
    helm install keda kedacore/keda --namespace keda --create-namespace
fi

# ── 5. Deploy CreditPulse ─────────────────────────────────
echo "[5/7] Creating namespace and secrets..."
kubectl apply -f infra/k8s/deployments.yaml

kubectl create secret generic creditpulse-secrets \
    --namespace=creditpulse \
    --from-literal=database-url="postgresql://creditpulse:creditpulse@creditpulse-postgres:5432/creditpulse" \
    --dry-run=client -o yaml | kubectl apply -f -

echo "[6/7] Applying KEDA ScaledObjects..."
kubectl apply -f infra/k8s/keda-scaledobject.yaml

# ── 6. Add hosts entry ────────────────────────────────────
MINIKUBE_IP=$(minikube ip)
if ! grep -q "creditpulse.local" /etc/hosts; then
    echo "[7/7] Adding /etc/hosts entry (requires sudo)..."
    echo "$MINIKUBE_IP creditpulse.local" | sudo tee -a /etc/hosts
else
    echo "[7/7] /etc/hosts already has creditpulse.local ✓"
fi

# ── 7. Summary ────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " CreditPulse deployed to minikube!"
echo ""
echo "  Dashboard  →  http://creditpulse.local"
echo "  API docs   →  http://creditpulse.local/api/docs"
echo "  MLflow     →  http://$(minikube ip):30500"
echo ""
echo "  kubectl get pods -n creditpulse"
echo "  kubectl logs -n creditpulse deploy/creditpulse-api -f"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
