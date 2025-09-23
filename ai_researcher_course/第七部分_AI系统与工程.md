# 第七部分：AI系统与工程

## 第19章 模型部署与服务化

### 19.1 模型部署基础

#### 19.1.1 部署环境概述

**部署环境的类型**：
```
本地部署：单机环境，适合开发和测试
服务器部署：企业级服务器，集中式管理
云端部署：云平台服务，弹性扩展
边缘部署：靠近数据源，低延迟
```

**部署架构选择**：
- **单体架构**：简单直接，易于维护
- **微服务架构**：模块化，易于扩展
- **无服务器架构**：按需使用，成本优化
- **混合架构**：结合多种部署方式

**部署策略**：
```
蓝绿部署：零停机时间，快速回滚
金丝雀部署：渐进式发布，风险控制
A/B测试部署：对比验证，数据驱动
影子部署：并行运行，验证效果
```

#### 19.1.2 模型序列化与保存

**模型格式选择**：
```
Pickle：Python原生格式，简单易用
Joblib：适合大数据对象，压缩效率高
ONNX：开放标准，跨平台兼容
PMML：预测模型标记语言，标准化
```

**模型序列化最佳实践**：
```python
# 保存完整模型
import joblib
joblib.dump(model, 'model.pkl')

# 保存模型权重
model.save_weights('model_weights.h5')

# 保存模型架构
model_json = model.to_json()
with open('model_architecture.json', 'w') as f:
    f.write(model_json)
```

**版本控制**：
```
模型版本：语义化版本控制（v1.0.0）
数据版本：训练数据版本管理
代码版本：模型代码版本控制
配置版本：超参数和配置版本
```

#### 19.1.3 模型优化技术

**量化技术**：
```
8位整数量化：减少75%内存占用
16位浮点量化：平衡精度和性能
动态量化：运行时量化，灵活性强
静态量化：预量化，推理速度快
```

**剪枝技术**：
```
结构化剪枝：移除整个通道或层
非结构化剪枝：移除单个权重
渐进式剪枝：逐步增加剪枝比例
敏感性分析：基于重要性剪枝
```

**知识蒸馏**：
```
教师-学生架构：大模型指导小模型
蒸馏损失函数：匹配输出分布
中间层匹配：匹配隐藏层表示
多教师集成：多个教师指导
```

**编译优化**：
```
图优化：计算图优化和简化
算子融合：合并连续操作
内存优化：减少内存分配
并行化：充分利用硬件资源
```

### 19.2 推理服务架构

#### 19.2.1 Web服务框架

**Flask框架**：
```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict(data['features'])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**FastAPI框架**：
```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load('model.pkl')

class PredictionRequest(BaseModel):
    features: list

@app.post('/predict')
async def predict(request: PredictionRequest):
    prediction = model.predict([request.features])
    return {'prediction': prediction[0]}
```

**Docker容器化**：
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY model.pkl .
COPY app.py .

EXPOSE 5000
CMD ['python', 'app.py']
```

#### 19.2.2 高性能推理引擎

**TensorFlow Serving**：
```
特点：专为TensorFlow模型设计
优势：高性能，版本管理，监控
部署：Docker容器，易于扩展
API：gRPC和REST API支持
```

**TorchServe**：
```
特点：PyTorch官方服务框架
功能：模型管理，日志记录，指标监控
支持：多模型，批处理，动态批处理
扩展：自定义处理逻辑，中间件
```

**ONNX Runtime**：
```
跨平台：支持多种硬件和操作系统
性能优化：图优化，内存优化
格式支持：ONNX格式模型
硬件加速：CPU，GPU，TPU支持
```

**Triton推理服务器**：
```
多框架支持：TensorFlow，PyTorch，ONNX等
并发处理：动态批处理，模型流水线
性能优化：内存管理，计算优化
可扩展性：水平扩展，负载均衡
```

#### 19.2.3 异步处理与批处理

**异步推理**：
```python
import asyncio
import aiohttp

async def async_predict(session, data):
    async with session.post('http://localhost:5000/predict', json=data) as response:
        return await response.json()

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [async_predict(session, data) for data in input_data]
        results = await asyncio.gather(*tasks)
        return results
```

**动态批处理**：
```python
import time
from collections import deque

class DynamicBatcher:
    def __init__(self, max_batch_size=32, max_wait_time=0.1):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.batch_queue = deque()
        self.last_batch_time = time.time()

    def add_request(self, request):
        self.batch_queue.append(request)
        if len(self.batch_queue) >= self.max_batch_size or \
           time.time() - self.last_batch_time >= self.max_wait_time:
            return self.process_batch()
        return None

    def process_batch(self):
        if not self.batch_queue:
            return None

        batch = list(self.batch_queue)
        self.batch_queue.clear()
        self.last_batch_time = time.time()

        # 处理批次数据
        batch_data = [req['data'] for req in batch]
        predictions = model.predict(batch_data)

        return [{'id': req['id'], 'prediction': pred}
                for req, pred in zip(batch, predictions)]
```

**请求调度**：
```
轮询调度：简单公平的负载分配
加权轮询：根据服务器性能分配
最少连接：优先分配给负载轻的服务器
一致性哈希：保证会话亲和性
```

### 19.3 监控与维护

#### 19.3.1 性能监控

**关键指标监控**：
```
响应时间：API响应延迟
吞吐量：每秒处理请求数
错误率：请求失败比例
资源利用率：CPU、内存、GPU使用率
```

**Prometheus监控**：
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ml_service'
    static_configs:
      - targets: ['localhost:8000']
```

```python
# 应用监控指标
from prometheus_client import Counter, Histogram, Gauge

# 定义指标
REQUEST_COUNT = Counter('requests_total', 'Total requests')
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')
ACTIVE_REQUESTS = Gauge('active_requests', 'Number of active requests')

@app.route('/predict')
@REQUEST_DURATION.time()
def predict():
    ACTIVE_REQUESTS.inc()
    REQUEST_COUNT.inc()
    try:
        # 预测逻辑
        result = model.predict(data)
        return jsonify(result)
    finally:
        ACTIVE_REQUESTS.dec()
```

**日志监控**：
```python
import logging
import json
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_service.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def log_prediction(input_data, prediction, latency):
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'input_shape': input_data.shape,
        'prediction': prediction.tolist(),
        'latency_ms': latency * 1000
    }
    logger.info(json.dumps(log_entry))
```

#### 19.3.2 模型漂移检测

**数据漂移检测**：
```python
import numpy as np
from scipy import stats

def detect_data_drift(reference_data, current_data, threshold=0.05):
    drift_detected = False
    drift_features = []

    for feature in reference_data.columns:
        # Kolmogorov-Smirnov检验
        ks_stat, ks_pvalue = stats.ks_2samp(
            reference_data[feature],
            current_data[feature]
        )

        if ks_pvalue < threshold:
            drift_detected = True
            drift_features.append(feature)
            logger.warning(f"Data drift detected in feature {feature}: p-value={ks_pvalue}")

    return drift_detected, drift_features
```

**概念漂移检测**：
```python
def detect_concept_drift(model, reference_data, current_data, threshold=0.1):
    # 在参考数据上的性能
    ref_predictions = model.predict(reference_data.drop('target', axis=1))
    ref_accuracy = accuracy_score(reference_data['target'], ref_predictions)

    # 在当前数据上的性能
    curr_predictions = model.predict(current_data.drop('target', axis=1))
    curr_accuracy = accuracy_score(current_data['target'], curr_predictions)

    performance_drop = ref_accuracy - curr_accuracy

    if performance_drop > threshold:
        logger.warning(f"Concept drift detected: performance drop={performance_drop}")
        return True, performance_drop

    return False, performance_drop
```

**在线监控**：
```python
class ModelMonitor:
    def __init__(self, model, reference_data, check_interval=3600):
        self.model = model
        self.reference_data = reference_data
        self.check_interval = check_interval
        self.last_check = time.time()

    def check_predictions(self, input_data, predictions):
        current_time = time.time()

        # 定期检查漂移
        if current_time - self.last_check > self.check_interval:
            self.run_drift_detection(input_data)
            self.last_check = current_time

        # 实时监控预测分布
        self.monitor_prediction_distribution(predictions)

    def run_drift_detection(self, current_data):
        # 执行漂移检测
        data_drift, data_features = detect_data_drift(
            self.reference_data, current_data
        )

        concept_drift, performance_drop = detect_concept_drift(
            self.model, self.reference_data, current_data
        )

        if data_drift or concept_drift:
            self.trigger_alert(data_drift, concept_drift)

    def trigger_alert(self, data_drift, concept_drift):
        # 发送告警
        alert_message = f"ALERT: Data drift={data_drift}, Concept drift={concept_drift}"
        logger.critical(alert_message)

        # 可以集成邮件、短信等告警机制
        send_alert_email(alert_message)
```

#### 19.3.3 自动化维护

**自动重启动机制**：
```python
import subprocess
import time

class ServiceHealthChecker:
    def __init__(self, service_url, restart_command):
        self.service_url = service_url
        self.restart_command = restart_command

    def check_health(self):
        try:
            response = requests.get(f"{self.service_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def restart_service(self):
        logger.warning("Service unhealthy, restarting...")
        try:
            subprocess.run(self.restart_command, shell=True, check=True)
            logger.info("Service restarted successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to restart service: {e}")

    def monitor(self, check_interval=30):
        while True:
            if not self.check_health():
                self.restart_service()
            time.sleep(check_interval)
```

**自动扩缩容**：
```python
import kubernetes
from kubernetes import client, config

class AutoScaler:
    def __init__(self, deployment_name, namespace='default'):
        config.load_kube_config()
        self.apps_v1 = client.AppsV1Api()
        self.deployment_name = deployment_name
        self.namespace = namespace

    def get_current_replicas(self):
        deployment = self.apps_v1.read_namespaced_deployment(
            self.deployment_name, self.namespace
        )
        return deployment.spec.replicas

    def scale_deployment(self, replicas):
        deployment = self.apps_v1.read_namespaced_deployment(
            self.deployment_name, self.namespace
        )
        deployment.spec.replicas = replicas

        self.apps_v1.patch_namespaced_deployment(
            self.deployment_name,
            self.namespace,
            deployment
        )
        logger.info(f"Scaled deployment to {replicas} replicas")

    def auto_scale(self, cpu_threshold=70, min_replicas=1, max_replicas=10):
        # 获取当前CPU使用率
        current_cpu = self.get_cpu_usage()
        current_replicas = self.get_current_replicas()

        if current_cpu > cpu_threshold and current_replicas < max_replicas:
            # 扩容
            new_replicas = min(current_replicas + 1, max_replicas)
            self.scale_deployment(new_replicas)
        elif current_cpu < cpu_threshold * 0.5 and current_replicas > min_replicas:
            # 缩容
            new_replicas = max(current_replicas - 1, min_replicas)
            self.scale_deployment(new_replicas)
```

**模型自动更新**：
```python
import shutil
import hashlib

class ModelUpdater:
    def __init__(self, model_path, backup_path='./backups'):
        self.model_path = model_path
        self.backup_path = backup_path
        self.current_hash = self.calculate_model_hash()

    def calculate_model_hash(self):
        with open(self.model_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def backup_model(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(self.backup_path, f"model_{timestamp}.pkl")
        shutil.copy2(self.model_path, backup_file)
        logger.info(f"Model backed up to {backup_file}")
        return backup_file

    def check_model_update(self):
        new_hash = self.calculate_model_hash()
        if new_hash != self.current_hash:
            logger.info("Model update detected")
            backup_file = self.backup_model()
            self.reload_model()
            self.current_hash = new_hash
            return True
        return False

    def reload_model(self):
        # 重新加载模型
        global model
        model = joblib.load(self.model_path)
        logger.info("Model reloaded successfully")
```

## 第20章 MLOps与DevOps

### 20.1 MLOps概述

#### 20.1.1 MLOps的定义与价值

**MLOps的定义**：
MLOps（Machine Learning Operations）是机器学习系统的DevOps实践，旨在标准化和自动化ML模型的整个生命周期管理。

**MLOps的核心价值**：
```
自动化：减少人工干预，提高效率
可重现性：确保实验结果可重现
可靠性：提高系统稳定性和可靠性
可扩展性：支持大规模模型部署和管理
```

**MLOps与传统DevOps的区别**：
```
数据管理：MLOps需要处理数据版本和漂移
模型管理：MLOps涉及模型训练、评估和部署
实验跟踪：MLOps需要跟踪实验和超参数
监控：MLOps需要监控模型性能和数据漂移
```

#### 20.1.2 MLOps成熟度模型

**级别0：手动流程**：
```
特点：完全手动，没有自动化
挑战：难以重现，效率低下
适用：小型项目，概念验证
```

**级别1：ML管道自动化**：
```
特点：训练管道自动化
组件：数据准备、模型训练、评估
工具：Airflow、Kubeflow Pipelines
```

**级别2：CI/CD/CT自动化**：
```
CI：持续集成，代码和模型集成测试
CD：持续部署，自动化模型部署
CT：持续训练，自动化模型再训练
```

**级别3：完全自动化**：
```
特点：端到端自动化监控和反馈
功能：自动模型更新、回滚、扩缩容
工具：完整的MLOps平台
```

#### 20.1.3 MLOps工具栈

**数据管理工具**：
```
DVC：数据版本控制
Delta Lake：事务性数据湖
Great Expectations：数据质量验证
Apache Atlas：数据治理和元数据管理
```

**实验跟踪工具**：
```
MLflow：实验跟踪和模型管理
Weights & Biases：实验可视化和协作
TensorBoard：TensorFlow实验可视化
Neptune：实验管理平台
```

**模型部署工具**：
```
Seldon Core：Kubernetes上的模型部署
KServe：无服务器模型推理平台
BentoML：Python模型服务框架
MLflow Model Serving：MLflow模型服务
```

**监控工具**：
```
Evidently AI：模型性能监控
WhyLogs：数据监控和验证
Arize：模型可观测性平台
Fiddler：模型监控和解释
```

### 20.2 CI/CD管道

#### 20.2.1 持续集成（CI）

**CI管道的主要步骤**：
```
1. 代码提交触发
2. 依赖安装
3. 单元测试
4. 代码质量检查
5. 数据验证
6. 模型训练
7. 模型评估
8. 构建 artifacts
```

**GitHub Actions CI配置**：
```yaml
name: ML CI Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Run tests
      run: |
        python -m pytest tests/

    - name: Data validation
      run: |
        python scripts/validate_data.py

    - name: Model training
      run: |
        python scripts/train.py

    - name: Model evaluation
      run: |
        python scripts/evaluate.py
```

**GitLab CI配置**：
```yaml
stages:
  - test
  - train
  - evaluate

variables:
  PYTHON_VERSION: "3.9"

before_script:
  - pip install -r requirements.txt

test_job:
  stage: test
  script:
    - python -m pytest tests/
    - python scripts/validate_data.py

train_job:
  stage: train
  script:
    - python scripts/train.py
  artifacts:
    paths:
      - models/

evaluate_job:
  stage: evaluate
  script:
    - python scripts/evaluate.py
  dependencies:
    - train_job
```

#### 20.2.2 持续部署（CD）

**CD管道配置**：
```yaml
name: ML CD Pipeline

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Download model artifact
      uses: actions/download-artifact@v2
      with:
        name: model-artifact

    - name: Build and push Docker image
      run: |
        docker build -t ml-service:${{ github.sha }} .
        docker push ml-service:${{ github.sha }}

    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/ml-service \
          ml-service=ml-service:${{ github.sha }}
```

**蓝绿部署策略**：
```python
import kubernetes
from kubernetes import client, config

class BlueGreenDeployer:
    def __init__(self, app_name, namespace='default'):
        config.load_kube_config()
        self.apps_v1 = client.AppsV1Api()
        self.core_v1 = client.CoreV1Api()
        self.app_name = app_name
        self.namespace = namespace

    def deploy_new_version(self, new_image):
        # 创建绿色环境
        green_deployment = self.create_green_deployment(new_image)

        # 等待绿色环境就绪
        self.wait_for_deployment_ready(f"{self.app_name}-green")

        # 切换流量
        self.switch_traffic(f"{self.app_name}-green")

        # 删除旧环境
        self.delete_deployment(f"{self.app_name}-blue")

        # 重命名绿色为蓝色
        self.rename_deployment(f"{self.app_name}-green", f"{self.app_name}-blue")

    def create_green_deployment(self, image):
        # 创建绿色部署配置
        deployment = client.V1Deployment(
            metadata=client.V1ObjectMeta(
                name=f"{self.app_name}-green",
                labels={"app": self.app_name, "version": "green"}
            ),
            spec=client.V1DeploymentSpec(
                replicas=3,
                selector=client.V1LabelSelector(
                    match_labels={"app": self.app_name, "version": "green"}
                ),
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={"app": self.app_name, "version": "green"}
                    ),
                    spec=client.V1PodSpec(
                        containers=[client.V1Container(
                            name="ml-service",
                            image=image,
                            ports=[client.V1ContainerPort(container_port=5000)]
                        )]
                    )
                )
            )
        )

        return self.apps_v1.create_namespaced_deployment(
            namespace=self.namespace,
            body=deployment
        )
```

#### 20.2.3 持续训练（CT）

**自动重训练触发器**：
```python
import schedule
import time
from datetime import datetime, timedelta

class ContinuousTrainer:
    def __init__(self, retrain_config):
        self.config = retrain_config
        self.last_retrain = datetime.now()

    def check_retrain_conditions(self):
        conditions = []

        # 时间触发
        if datetime.now() - self.last_retrain > timedelta(days=self.config['retrain_interval']):
            conditions.append('schedule')

        # 性能下降触发
        if self.check_performance_degradation():
            conditions.append('performance')

        # 数据漂移触发
        if self.check_data_drift():
            conditions.append('data_drift')

        return conditions

    def check_performance_degradation(self):
        # 检查模型性能是否下降
        current_performance = self.get_current_performance()
        baseline_performance = self.get_baseline_performance()

        degradation_threshold = self.config['performance_threshold']
        return current_performance < baseline_performance * (1 - degradation_threshold)

    def check_data_drift(self):
        # 检查数据漂移
        return detect_data_drift(self.reference_data, self.current_data)[0]

    def trigger_retraining(self, trigger_reasons):
        logger.info(f"Triggering retraining due to: {trigger_reasons}")

        # 执行重训练
        new_model = self.retrain_model()

        # 评估新模型
        evaluation_results = self.evaluate_model(new_model)

        # 如果性能更好，部署新模型
        if self.should_deploy(new_model, evaluation_results):
            self.deploy_model(new_model)
            self.last_retrain = datetime.now()
            logger.info("Model retrained and deployed successfully")
        else:
            logger.info("New model performance not sufficient, keeping current model")

    def schedule_retraining_checks(self):
        # 每小时检查一次重训练条件
        schedule.every().hour.do(self.retraining_check)

        while True:
            schedule.run_pending()
            time.sleep(60)

    def retraining_check(self):
        trigger_reasons = self.check_retrain_conditions()
        if trigger_reasons:
            self.trigger_retraining(trigger_reasons)
```

### 20.3 实验管理与版本控制

#### 20.3.1 实验跟踪

**MLflow实验跟踪**：
```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model_with_mlflow(params, train_data, val_data):
    with mlflow.start_run():
        # 记录参数
        mlflow.log_params(params)

        # 训练模型
        model = RandomForestClassifier(**params)
        model.fit(train_data['X'], train_data['y'])

        # 预测和评估
        predictions = model.predict(val_data['X'])
        accuracy = accuracy_score(val_data['y'], predictions)

        # 记录指标
        mlflow.log_metric('accuracy', accuracy)

        # 记录模型
        mlflow.sklearn.log_model(model, 'model')

        # 记录额外信息
        mlflow.log_artifact('preprocessing_pipeline.pkl')
        mlflow.log_dict('data_stats.json', calculate_data_stats(train_data))

        return model, accuracy
```

**自定义实验跟踪**：
```python
import json
import pandas as pd
from datetime import datetime
import sqlite3

class ExperimentTracker:
    def __init__(self, db_path='experiments.db'):
        self.conn = sqlite3.connect(db_path)
        self.setup_database()

    def setup_database(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                parameters TEXT,
                metrics TEXT,
                artifacts TEXT,
                status TEXT
            )
        ''')
        self.conn.commit()

    def start_experiment(self, name, parameters):
        experiment_id = self.log_experiment_start(name, parameters)
        return experiment_id

    def log_experiment_start(self, name, parameters):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO experiments (name, parameters, status)
            VALUES (?, ?, ?)
        ''', (name, json.dumps(parameters), 'running'))
        self.conn.commit()
        return cursor.lastrowid

    def log_metrics(self, experiment_id, metrics):
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE experiments
            SET metrics = ?, status = 'completed'
            WHERE id = ?
        ''', (json.dumps(metrics), experiment_id))
        self.conn.commit()

    def log_artifacts(self, experiment_id, artifacts):
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE experiments
            SET artifacts = ?
            WHERE id = ?
        ''', (json.dumps(artifacts), experiment_id))
        self.conn.commit()

    def get_experiment_history(self, limit=10):
        query = '''
            SELECT id, name, timestamp, parameters, metrics, status
            FROM experiments
            ORDER BY timestamp DESC
            LIMIT ?
        '''
        df = pd.read_sql_query(query, self.conn, params=(limit,))
        return df

    def get_best_experiment(self, metric_name='accuracy'):
        query = '''
            SELECT id, name, timestamp, parameters, metrics
            FROM experiments
            WHERE status = 'completed'
            ORDER BY json_extract(metrics, '$.{}') DESC
            LIMIT 1
        '''.format(metric_name)

        cursor = self.conn.cursor()
        cursor.execute(query)
        result = cursor.fetchone()

        if result:
            return {
                'id': result[0],
                'name': result[1],
                'timestamp': result[2],
                'parameters': json.loads(result[3]),
                'metrics': json.loads(result[4])
            }
        return None
```

#### 20.3.2 模型版本控制

**DVC数据版本控制**：
```bash
# 初始化DVC
dvc init

# 添加数据文件到DVC
dvc add data/raw/dataset.csv

# 添加模型文件到DVC
dvc add models/random_forest.pkl

# 推送到远程存储
dvc remote add -d myremote s3://my-bucket/dvc-store
dvc push

# 检查数据变化
dvc status
```

**模型版本管理**：
```python
import os
import shutil
import json
from datetime import datetime

class ModelVersionManager:
    def __init__(self, model_registry_path='./model_registry'):
        self.registry_path = model_registry_path
        self.current_version_path = os.path.join(model_registry_path, 'current_version.json')
        self.setup_registry()

    def setup_registry(self):
        os.makedirs(self.registry_path, exist_ok=True)

        # 初始化当前版本文件
        if not os.path.exists(self.current_version_path):
            with open(self.current_version_path, 'w') as f:
                json.dump({'current_version': '0.0.0'}, f)

    def register_model(self, model_path, metrics, parameters, version=None):
        if version is None:
            version = self.get_next_version()

        # 创建版本目录
        version_dir = os.path.join(self.registry_path, f'v{version}')
        os.makedirs(version_dir, exist_ok=True)

        # 复制模型文件
        model_filename = os.path.basename(model_path)
        version_model_path = os.path.join(version_dir, model_filename)
        shutil.copy2(model_path, version_model_path)

        # 创建版本元数据
        metadata = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'parameters': parameters,
            'model_path': version_model_path
        }

        metadata_path = os.path.join(version_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # 更新当前版本
        self.update_current_version(version)

        logger.info(f"Model version {version} registered successfully")
        return version

    def get_next_version(self):
        current_version = self.get_current_version()
        major, minor, patch = current_version.split('.')
        return f"{major}.{minor}.{int(patch) + 1}"

    def get_current_version(self):
        with open(self.current_version_path, 'r') as f:
            data = json.load(f)
            return data['current_version']

    def update_current_version(self, version):
        with open(self.current_version_path, 'w') as f:
            json.dump({'current_version': version}, f)

    def load_model(self, version=None):
        if version is None:
            version = self.get_current_version()

        version_dir = os.path.join(self.registry_path, f'v{version}')
        metadata_path = os.path.join(version_dir, 'metadata.json')

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        model = joblib.load(metadata['model_path'])
        return model, metadata

    def list_versions(self):
        versions = []
        for item in os.listdir(self.registry_path):
            if item.startswith('v'):
                version = item[1:]  # Remove 'v' prefix
                metadata_path = os.path.join(self.registry_path, item, 'metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    versions.append(metadata)

        return sorted(versions, key=lambda x: x['version'])

    def rollback(self, target_version):
        version_dir = os.path.join(self.registry_path, f'v{target_version}')
        if not os.path.exists(version_dir):
            raise ValueError(f"Version {target_version} not found")

        self.update_current_version(target_version)
        logger.info(f"Rolled back to version {target_version}")
```

#### 20.3.3 配置管理

**配置文件管理**：
```python
import yaml
import os
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config_path='config'):
        self.config_path = config_path
        self.base_config = self.load_config('base.yaml')
        self.environment_config = self.load_config(f'{os.getenv("ENV", "development")}.yaml')
        self.merged_config = self.merge_configs(self.base_config, self.environment_config)

    def load_config(self, filename):
        file_path = os.path.join(self.config_path, filename)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def merge_configs(self, base, override):
        merged = base.copy()
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self.merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged

    def get_config(self, key_path=None):
        if key_path is None:
            return self.merged_config

        keys = key_path.split('.')
        value = self.merged_config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                raise KeyError(f"Key '{key}' not found in config")

        return value

    def update_config(self, key_path, value):
        keys = key_path.split('.')
        config = self.merged_config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value
        logger.info(f"Config updated: {key_path} = {value}")
```

**环境变量管理**：
```python
import os
from dotenv import load_dotenv

class EnvironmentManager:
    def __init__(self):
        load_dotenv()

    def get_required_env(self, key):
        value = os.getenv(key)
        if value is None:
            raise ValueError(f"Required environment variable '{key}' not found")
        return value

    def get_env_with_default(self, key, default):
        return os.getenv(key, default)

    def validate_env_vars(self, required_vars):
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")

    def setup_mlflow_tracking(self):
        mlflow_tracking_uri = self.get_env_with_default('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(self.get_env_with_default('MLFLOW_EXPERIMENT_NAME', 'default'))

    def setup_database_connection(self):
        db_config = {
            'host': self.get_required_env('DB_HOST'),
            'port': self.get_env_with_default('DB_PORT', '5432'),
            'database': self.get_required_env('DB_NAME'),
            'username': self.get_required_env('DB_USER'),
            'password': self.get_required_env('DB_PASSWORD')
        }
        return db_config
```

**超参数管理**：
```python
import optuna
import json

class HyperparameterManager:
    def __init__(self, study_name='ml_optimization'):
        self.study_name = study_name
        self.study = None

    def create_study(self, direction='maximize', storage=None):
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction=direction,
            storage=storage,
            load_if_exists=True
        )
        return self.study

    def objective_function(self, trial, train_func, eval_func):
        # 定义搜索空间
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 200),
            'max_depth': trial.suggest_int('max_depth', 1, 20),
            'min_samples_split': trial.suggest_float('min_samples_split', 0.1, 1.0),
            'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.1, 0.5),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
        }

        # 训练模型
        model = train_func(params)

        # 评估模型
        score = eval_func(model)

        return score

    def optimize(self, train_func, eval_func, n_trials=100):
        if self.study is None:
            self.create_study()

        def objective(trial):
            return self.objective_function(trial, train_func, eval_func)

        self.study.optimize(objective, n_trials=n_trials)
        return self.study.best_params, self.study.best_value

    def save_study(self, file_path):
        if self.study:
            trials = self.study.trials
            study_data = {
                'best_params': self.study.best_params,
                'best_value': self.study.best_value,
                'trials': [
                    {
                        'params': trial.params,
                        'value': trial.value,
                        'state': trial.state.name
                    }
                    for trial in trials
                ]
            }

            with open(file_path, 'w') as f:
                json.dump(study_data, f, indent=2)

    def load_study(self, file_path):
        with open(file_path, 'r') as f:
            study_data = json.load(f)

        self.create_study()
        self.study.best_params = study_data['best_params']
        self.study.best_value = study_data['best_value']

        return study_data
```

## 第21章 分布式系统与性能优化

### 21.1 分布式计算基础

#### 21.1.1 分布式系统架构

**分布式系统的特点**：
```
资源分布：计算和存储资源分布在多个节点
并行处理：多个任务并行执行
容错性：节点故障时系统仍能正常运行
可扩展性：可以根据需求添加或删除节点
```

**架构模式**：
```
主从架构：主节点负责任务分配，从节点执行任务
对等架构：所有节点地位平等，相互协作
分层架构：按功能分层，每层负责特定功能
微服务架构：应用拆分为多个独立服务
```

**通信模式**：
```
同步通信：等待响应，阻塞式
异步通信：不等待响应，非阻塞式
发布订阅：消息队列，解耦生产者和消费者
远程过程调用：像调用本地函数一样调用远程服务
```

#### 21.1.2 分布式文件系统

**HDFS架构**：
```
NameNode：管理文件系统元数据
DataNode：存储实际数据块
客户端：访问文件系统的接口
Secondary NameNode：辅助NameNode，定期合并编辑日志
```

**HDFS操作**：
```python
from hdfs import InsecureClient

class HDFSManager:
    def __init__(self, namenode_url):
        self.client = InsecureClient(namenode_url)

    def upload_file(self, local_path, hdfs_path):
        with self.client.write(hdfs_path) as writer:
            with open(local_path, 'rb') as reader:
                writer.write(reader.read())

    def download_file(self, hdfs_path, local_path):
        with self.client.read(hdfs_path) as reader:
            with open(local_path, 'wb') as writer:
                writer.write(reader.read())

    def list_directory(self, hdfs_path):
        return self.client.list(hdfs_path)

    def make_directory(self, hdfs_path):
        self.client.makedirs(hdfs_path)

    def delete_path(self, hdfs_path, recursive=False):
        self.client.delete(hdfs_path, recursive=recursive)
```

**对象存储**：
```python
import boto3

class S3Manager:
    def __init__(self, access_key, secret_key, region):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )

    def upload_file(self, local_path, bucket_name, s3_key):
        self.s3_client.upload_file(local_path, bucket_name, s3_key)

    def download_file(self, bucket_name, s3_key, local_path):
        self.s3_client.download_file(bucket_name, s3_key, local_path)

    def list_objects(self, bucket_name, prefix=''):
        response = self.s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix
        )
        return response.get('Contents', [])

    def copy_object(self, source_bucket, source_key, dest_bucket, dest_key):
        copy_source = {'Bucket': source_bucket, 'Key': source_key}
        self.s3_client.copy_object(
            CopySource=copy_source,
            Bucket=dest_bucket,
            Key=dest_key
        )
```

#### 21.1.3 分布式协调服务

**ZooKeeper基础操作**：
```python
from kazoo.client import KazooClient

class ZooKeeperManager:
    def __init__(self, hosts):
        self.zk = KazooClient(hosts=hosts)
        self.zk.start()

    def create_node(self, path, value=None, ephemeral=False, sequence=False):
        return self.zk.create(
            path,
            value=value.encode() if value else None,
            ephemeral=ephemeral,
            sequence=sequence
        )

    def get_node(self, path, watch=None):
        data, stat = self.zk.get(path, watch=watch)
        return data.decode() if data else None, stat

    def set_node(self, path, value):
        self.zk.set(path, value.encode())

    def delete_node(self, path, recursive=False):
        self.zk.delete(path, recursive=recursive)

    def get_children(self, path, watch=None):
        return self.zk.get_children(path, watch=watch)

    def exists(self, path, watch=None):
        return self.zk.exists(path, watch=watch)

    def create_lock(self, path):
        return self.zk.Lock(path)

    def create_election(self, path):
        return self.zk.Election(path)

    def close(self):
        self.zk.stop()
        self.zk.close()
```

**分布式锁实现**：
```python
import time
import uuid
from threading import Lock

class DistributedLock:
    def __init__(self, zk_manager, lock_path):
        self.zk = zk_manager
        self.lock_path = lock_path
        self.local_lock = Lock()
        self.lock_id = str(uuid.uuid4())

    def acquire(self, timeout=30):
        start_time = time.time()

        while time.time() - start_time < timeout:
            with self.local_lock:
                try:
                    # 尝试创建临时有序节点
                    lock_node = self.zk.create_node(
                        f"{self.lock_path}/lock-",
                        value=self.lock_id,
                        ephemeral=True,
                        sequence=True
                    )

                    # 检查是否是最小的节点
                    children = self.zk.get_children(self.lock_path)
                    children.sort()

                    if lock_node.split('/')[-1] == children[0]:
                        return True

                    # 监听前一个节点
                    index = children.index(lock_node.split('/')[-1])
                    if index > 0:
                        prev_node = f"{self.lock_path}/{children[index-1]}"

                        def watch_prev_node(event):
                            pass

                        self.zk.exists(prev_node, watch=watch_prev_node)
                        time.sleep(1)
                        continue

                except Exception as e:
                    logger.error(f"Failed to acquire lock: {e}")
                    time.sleep(1)

        return False

    def release(self):
        try:
            # 删除自己的锁节点
            children = self.zk.get_children(self.lock_path)
            for child in children:
                node_path = f"{self.lock_path}/{child}"
                data, _ = self.zk.get_node(node_path)
                if data == self.lock_id:
                    self.zk.delete_node(node_path)
                    break
        except Exception as e:
            logger.error(f"Failed to release lock: {e}")
```

### 21.2 分布式机器学习

#### 21.2.1 数据并行

**PyTorch数据并行**：
```python
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_process(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # 初始化进程组
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )

def cleanup():
    dist.destroy_process_group()

class DistributedTrainer:
    def __init__(self, model, train_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device

        # 包装模型为DDP
        self.model = DDP(self.model, device_ids=[device])

    def train_epoch(self, optimizer, criterion, epoch):
        self.model.train()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                      f'({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def train(rank, world_size, model_class, train_dataset):
    setup_process(rank, world_size)

    # 创建数据加载器
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        sampler=train_sampler,
        num_workers=4
    )

    # 创建模型
    model = model_class().to(rank)
    model = DDP(model, device_ids=[rank])

    # 优化器和损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # 训练
    trainer = DistributedTrainer(model, train_loader, rank)

    for epoch in range(1, 11):
        train_sampler.set_epoch(epoch)
        trainer.train_epoch(optimizer, criterion, epoch)

    cleanup()

def main():
    world_size = torch.cuda.device_count()
    model_class = SimpleModel  # 你的模型类
    train_dataset = YourDataset()  # 你的数据集

    mp.spawn(train,
            args=(world_size, model_class, train_dataset),
            nprocs=world_size,
            join=True)
```

**TensorFlow分布式策略**：
```python
import tensorflow as tf

def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

def train_distributed():
    # 创建分布式策略
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        # 在策略作用域内创建模型
        model = create_model()
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    # 准备数据集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0

    # 创建分布式数据集
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(10000).batch(64)
    distributed_dataset = strategy.experimental_distribute_dataset(dataset)

    # 训练模型
    model.fit(distributed_dataset, epochs=10)

    return model

# 多工作器训练
def train_multi_worker():
    # 设置TF_CONFIG环境变量
    tf_config = {
        'cluster': {
            'worker': ['localhost:12345', 'localhost:12346']
        },
        'task': {'type': 'worker', 'index': 0}
    }

    os.environ['TF_CONFIG'] = json.dumps(tf_config)

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    with strategy.scope():
        model = create_model()
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    # 训练逻辑...
```

#### 21.2.2 模型并行

**模型并行实现**：
```python
import torch
import torch.nn as nn
import torch.distributed as dist

class ModelParallelModel(nn.Module):
    def __init__(self, rank, world_size):
        super(ModelParallelModel, self).__init__()
        self.rank = rank
        self.world_size = world_size

        # 将模型的不同层分配到不同的设备
        if rank == 0:
            # 第一个GPU处理前半部分
            self.part1 = nn.Sequential(
                nn.Linear(784, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU()
            ).to(rank)
        else:
            # 第二个GPU处理后半部分
            self.part2 = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            ).to(rank)

    def forward(self, x):
        if self.rank == 0:
            # 第一个GPU处理前向传播的前半部分
            x = self.part1(x)
            # 发送到第二个GPU
            dist.send(x, dst=1)
            # 接收第二个GPU的结果
            dist.recv(x, src=1)
            return x
        else:
            # 第二个GPU接收数据
            dist.recv(x, src=0)
            # 处理后半部分
            x = self.part2(x)
            # 发送回第一个GPU
            dist.send(x, dst=1)
            return x

class PipelineParallelModel(nn.Module):
    def __init__(self, rank, world_size):
        super(PipelineParallelModel, self).__init__()
        self.rank = rank
        self.world_size = world_size

        # 创建模型的各个阶段
        self.stages = nn.ModuleList([
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        ])

        # 将阶段分配到不同的设备
        stages_per_device = len(self.stages) // world_size
        self.local_stages = self.stages[
            rank * stages_per_device : (rank + 1) * stages_per_device
        ]

    def forward(self, x):
        # 处理本地阶段
        for stage in self.local_stages:
            x = stage(x)

        # 如果不是最后一个设备，发送到下一个设备
        if self.rank < self.world_size - 1:
            dist.send(x, dst=self.rank + 1)
        # 如果不是第一个设备，接收前一个设备的结果
        if self.rank > 0:
            dist.recv(x, src=self.rank - 1)

        return x
```

**混合并行（数据+模型并行）**：
```python
class HybridParallelModel(nn.Module):
    def __init__(self, rank, world_size, model_parallel_size):
        super(HybridParallelModel, self).__init__()

        self.rank = rank
        self.world_size = world_size
        self.model_parallel_size = model_parallel_size
        self.data_parallel_size = world_size // model_parallel_size

        # 计算当前进程在模型并行组中的位置
        self.model_parallel_rank = rank % model_parallel_size
        self.data_parallel_rank = rank // model_parallel_size

        # 创建模型并行组
        self.model_parallel_group = dist.new_group(
            ranks=list(range(self.model_parallel_rank,
                           world_size,
                           self.model_parallel_size))
        )

        # 创建数据并行组
        data_parallel_ranks = []
        for i in range(self.data_parallel_size):
            data_parallel_ranks.append(
                i * model_parallel_size + self.model_parallel_rank
            )
        self.data_parallel_group = dist.new_group(ranks=data_parallel_ranks)

        # 构建模型（模型并行）
        self.build_model()

    def build_model(self):
        # 根据模型并行rank构建模型的不同部分
        if self.model_parallel_rank == 0:
            self.encoder = nn.Sequential(
                nn.Linear(784, 512),
                nn.ReLU(),
                nn.Linear(512, 256)
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )

    def forward(self, x):
        # 模型并行前向传播
        if self.model_parallel_rank == 0:
            x = self.encoder(x)
            dist.send(x, dst=self.model_parallel_rank + 1)
        else:
            dist.recv(x, src=self.model_parallel_rank - 1)
            x = self.decoder(x)

        # 数据并行的all-reduce
        if self.model_parallel_rank == 1:  # 只有最后一个设备需要all-reduce
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=self.data_parallel_group)
            x = x / self.data_parallel_size

        return x
```

#### 21.2.3 参数服务器架构

**参数服务器实现**：
```python
import threading
import queue
import numpy as np

class ParameterServer:
    def __init__(self, model):
        self.model = model
        self.parameters = {name: param.data.numpy()
                         for name, param in model.named_parameters()}
        self.lock = threading.Lock()
        self.update_queue = queue.Queue()
        self.running = False

    def start(self):
        self.running = True
        self.server_thread = threading.Thread(target=self._run_server)
        self.server_thread.start()

    def stop(self):
        self.running = False
        self.server_thread.join()

    def _run_server(self):
        while self.running:
            try:
                # 从队列获取更新
                update = self.update_queue.get(timeout=1)

                # 应用更新
                with self.lock:
                    for name, grad in update.items():
                        if name in self.parameters:
                            self.parameters[name] -= 0.01 * grad  # 学习率0.01

            except queue.Empty:
                continue

    def get_parameters(self):
        with self.lock:
            return self.parameters.copy()

    def push_update(self, gradients):
        self.update_queue.put(gradients)

class Worker:
    def __init__(self, worker_id, parameter_server, train_data):
        self.worker_id = worker_id
        self.parameter_server = parameter_server
        self.train_data = train_data
        self.local_model = self.create_model()

    def create_model(self):
        # 创建本地模型副本
        model = YourModelClass()
        return model

    def train_step(self, data, target):
        # 获取最新参数
        parameters = self.parameter_server.get_parameters()

        # 更新本地模型参数
        with torch.no_grad():
            for name, param in self.local_model.named_parameters():
                if name in parameters:
                    param.data = torch.tensor(parameters[name])

        # 前向传播
        output = self.local_model(data)
        loss = nn.CrossEntropyLoss()(output, target)

        # 反向传播
        self.local_model.zero_grad()
        loss.backward()

        # 收集梯度
        gradients = {}
        for name, param in self.local_model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.numpy()

        # 发送梯度到参数服务器
        self.parameter_server.push_update(gradients)

        return loss.item()

    def train_epoch(self):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_data):
            loss = self.train_step(data, target)
            total_loss += loss

            if batch_idx % 100 == 0:
                print(f'Worker {self.worker_id} - Batch {batch_idx}, Loss: {loss:.4f}')

        return total_loss / len(self.train_data)

class DistributedTraining:
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.model = YourModelClass()
        self.parameter_server = ParameterServer(self.model)
        self.workers = []

    def setup_workers(self, train_datasets):
        for i in range(self.num_workers):
            worker = Worker(i, self.parameter_server, train_datasets[i])
            self.workers.append(worker)

    def train(self, epochs):
        # 启动参数服务器
        self.parameter_server.start()

        try:
            for epoch in range(epochs):
                print(f'Epoch {epoch + 1}/{epochs}')

                # 并行训练每个worker
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    futures = [executor.submit(worker.train_epoch) for worker in self.workers]
                    losses = [future.result() for future in concurrent.futures.as_completed(futures)]

                avg_loss = sum(losses) / len(losses)
                print(f'Average Loss: {avg_loss:.4f}')

        finally:
            # 停止参数服务器
            self.parameter_server.stop()
```

### 21.3 性能优化策略

#### 21.3.1 硬件加速

**GPU优化**：
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

class GPUOptimizedTrainer:
    def __init__(self, model, train_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.scaler = GradScaler()  # 混合精度训练

        # 使用CUDA优化
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

    def train_epoch_mixed_precision(self, optimizer, criterion, epoch):
        self.model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()

            # 使用自动混合精度
            with autocast():
                output = self.model(data)
                loss = criterion(output, target)

            # 缩放损失并反向传播
            self.scaler.scale(loss).backward()

            # 梯度裁剪
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 更新参数
            self.scaler.step(optimizer)
            self.scaler.update()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                      f'({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        return total_loss / len(self.train_loader)

    def optimize_memory_usage(self):
        # 使用梯度检查点
        from torch.utils.checkpoint import checkpoint

        class CheckpointedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(784, 512)
                self.layer2 = nn.Linear(512, 256)
                self.layer3 = nn.Linear(256, 10)

            def forward(self, x):
                x = checkpoint(self.layer1, x)
                x = checkpoint(self.layer2, x)
                x = self.layer3(x)
                return x

        # 使用内存高效的注意力
        if hasattr(torch.nn, 'functional'):
            torch.nn.functional.scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention

    def optimize_kernel_launch(self):
        # 使用CUDA图优化
        if torch.cuda.is_available():
            static_input = torch.randn(32, 784).to(self.device)
            static_target = torch.randint(0, 10, (32,)).to(self.device)

            # 预热
            for _ in range(10):
                self.model(static_input)

            # 创建CUDA图
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                static_output = self.model(static_input)

            # 重用CUDA图
            def run_with_graph(input_data):
                input_data.copy_(static_input)
                g.replay()
                return static_output.clone()

            self.model.forward = run_with_graph
```

**TPU优化**：
```python
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

class TPUOptimizedTrainer:
    def __init__(self, model, train_loader):
        self.device = xm.xla_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader

    def train_epoch_tpu(self, optimizer, criterion, epoch):
        self.model.train()
        total_loss = 0

        # 使用TPU优化的数据加载器
        para_loader = pl.ParallelLoader(
            self.train_loader,
            [self.device]
        ).per_device_loader(self.device)

        for batch_idx, (data, target) in enumerate(para_loader):
            optimizer.zero_grad()

            output = self.model(data)
            loss = criterion(output, target)

            loss.backward()

            # TPU特定的优化步骤
            xm.optimizer_step(optimizer)
            xm.mark_step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                # 打印日志（仅在主进程）
                if xm.is_master_ordinal():
                    print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                          f'({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        # 同步所有TPU核心
        total_loss = xm.mesh_reduce('loss', total_loss, torch.sum)

        return total_loss / len(self.train_loader)

    def optimize_tpu_memory(self):
        # 使用TPU内存优化
        import torch_xla.debug.profiler as xp

        # 内存分析
        xp.start_server(9012)

        # 使用TPU特定的优化
        if hasattr(torch_xla, 'optimization'):
            torch_xla.optimization.optimize_for_inference(self.model)
```

#### 21.3.2 算法优化

**模型压缩**：
```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

class ModelCompression:
    def __init__(self, model):
        self.model = model

    def prune_model(self, amount=0.2):
        # 对模型进行剪枝
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=amount)

        # 移除剪枝掩码
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                prune.remove(module, 'weight')

    def quantize_model(self):
        # 量化模型
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(self.model, inplace=True)

        # 校准（这里用一些示例数据）
        self.model.eval()
        with torch.no_grad():
            for _ in range(100):
                dummy_input = torch.randn(1, 3, 224, 224)
                self.model(dummy_input)

        # 转换为量化模型
        quantized_model = torch.quantization.convert(self.model, inplace=False)
        return quantized_model

    def knowledge_distillation(self, teacher_model, student_model, train_loader, temperature=4.0, alpha=0.7):
        # 知识蒸馏
        student_model.train()
        teacher_model.eval()

        optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
        criterion = nn.KLDivLoss(reduction='batchmean')
        hard_criterion = nn.CrossEntropyLoss()

        for epoch in range(10):
            total_loss = 0

            for data, target in train_loader:
                optimizer.zero_grad()

                # 教师模型的软标签
                with torch.no_grad():
                    teacher_output = teacher_model(data)
                    teacher_probs = torch.softmax(teacher_output / temperature, dim=1)

                # 学生模型的输出
                student_output = student_model(data)
                student_probs = torch.softmax(student_output / temperature, dim=1)

                # 计算蒸馏损失
                distill_loss = criterion(student_probs.log(), teacher_probs) * (temperature ** 2)
                hard_loss = hard_criterion(student_output, target)

                # 组合损失
                loss = alpha * distill_loss + (1 - alpha) * hard_loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f'Epoch {epoch}, Loss: {total_loss / len(train_loader):.4f}')

        return student_model
```

**计算图优化**：
```python
import torch
import torch.jit as jit

class GraphOptimizer:
    def __init__(self, model):
        self.model = model

    def torch_script_compile(self):
        # 转换为TorchScript
        scripted_model = torch.jit.script(self.model)
        return scripted_model

    def optimize_inference(self, model):
        # 优化推理
        model.eval()

        # 使用JIT编译
        if torch.cuda.is_available():
            model = torch.jit.optimize_for_inference(
                torch.jit.script(model)
            )

        return model

    def fuse_operations(self, model):
        # 融合操作
        torch.quantization.fuse_modules(model, [['conv', 'relu']], inplace=True)
        return model

    def memory_efficient_attention(self):
        # 内存高效的注意力机制
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            return torch.nn.functional.scaled_dot_product_attention
        else:
            # 回退到标准注意力
            def standard_attention(query, key, value):
                scores = torch.matmul(query, key.transpose(-2, -1))
                weights = torch.softmax(scores, dim=-1)
                return torch.matmul(weights, value)
            return standard_attention
```

#### 21.3.3 系统优化

**I/O优化**：
```python
import torch
from torch.utils.data import Dataset, DataLoader
import multiprocessing
import threading
import queue

class OptimizedDataLoader:
    def __init__(self, dataset, batch_size=32, num_workers=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers or multiprocessing.cpu_count()

        # 使用内存映射文件
        if hasattr(dataset, 'use_mmap'):
            dataset.use_mmap = True

        # 预取缓冲区
        self.prefetch_queue = queue.Queue(maxsize=2)
        self.prefetch_thread = None

    def start_prefetch(self):
        # 启动预取线程
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker)
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

    def _prefetch_worker(self):
        data_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True
        )

        for batch in data_loader:
            self.prefetch_queue.put(batch)

    def get_batch(self):
        return self.prefetch_queue.get()

class MemoryEfficientDataset(Dataset):
    def __init__(self, data_file, label_file, chunk_size=1000):
        self.data_file = data_file
        self.label_file = label_file
        self.chunk_size = chunk_size
        self.cache = {}

        # 使用内存映射
        self.data_mmap = np.memmap(data_file, dtype='float32', mode='r')
        self.label_mmap = np.memmap(label_file, dtype='int64', mode='r')

    def __len__(self):
        return len(self.label_mmap)

    def __getitem__(self, idx):
        # 计算chunk索引
        chunk_idx = idx // self.chunk_size
        chunk_offset = idx % self.chunk_size

        # 如果chunk不在缓存中，加载它
        if chunk_idx not in self.cache:
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, len(self))

            self.cache[chunk_idx] = {
                'data': torch.from_numpy(self.data_mmap[start_idx:end_idx]),
                'labels': torch.from_numpy(self.label_mmap[start_idx:end_idx])
            }

            # 限制缓存大小
            if len(self.cache) > 10:
                oldest_key = min(self.cache.keys())
                del self.cache[oldest_key]

        return self.cache[chunk_idx]['data'][chunk_offset], self.cache[chunk_idx]['labels'][chunk_offset]
```

**网络优化**：
```python
import socket
import threading
import queue
import numpy as np

class NetworkOptimizer:
    def __init__(self):
        self.compression_buffer = queue.Queue()
        self.compression_thread = None

    def start_compression(self):
        # 启动压缩线程
        self.compression_thread = threading.Thread(target=self._compress_worker)
        self.compression_thread.daemon = True
        self.compression_thread.start()

    def _compress_worker(self):
        while True:
            try:
                data = self.compression_buffer.get(timeout=1)
                compressed_data = self.compress_data(data)
                self.send_data(compressed_data)
            except queue.Empty:
                continue

    def compress_data(self, data, method='quantization'):
        if method == 'quantization':
            # 量化压缩
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()

            # 8-bit量化
            min_val, max_val = data.min(), data.max()
            scale = (max_val - min_val) / 255
            quantized = np.round((data - min_val) / scale).astype(np.uint8)

            return {
                'data': quantized,
                'min_val': min_val,
                'max_val': max_val,
                'shape': data.shape,
                'dtype': str(data.dtype)
            }

        elif method == 'sparsification':
            # 稀疏化
            threshold = np.percentile(np.abs(data), 90)
            mask = np.abs(data) > threshold
            sparse_data = data[mask]
            sparse_indices = np.where(mask)

            return {
                'data': sparse_data,
                'indices': sparse_indices,
                'shape': data.shape
            }

        return data

    def decompress_data(self, compressed_data):
        if 'min_val' in compressed_data:
            # 解量化
            quantized = compressed_data['data']
            min_val = compressed_data['min_val']
            max_val = compressed_data['max_val']
            shape = compressed_data['shape']

            scale = (max_val - min_val) / 255
            data = quantized.astype(np.float32) * scale + min_val
            return data.reshape(shape)

        elif 'indices' in compressed_data:
            # 解稀疏化
            shape = compressed_data['shape']
            data = np.zeros(shape)
            data[compressed_data['indices']] = compressed_data['data']
            return data

        return compressed_data

    def optimize_tcp_socket(self):
        # 优化TCP socket设置
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # 设置缓冲区大小
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)  # 1MB
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)  # 1MB

        # 启用TCP_NODELAY减少延迟
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        return sock

    def enable_gpu_direct(self):
        # GPU Direct RDMA
        if hasattr(torch, 'cuda'):
            torch.cuda.nvtx.mark_range_start("GPU Direct")
            # 这里可以添加GPU Direct的具体实现
            torch.cuda.nvtx.mark_range_end()
```

**缓存优化**：
```python
import functools
import time
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.cache:
                # 移到最后（最近使用）
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None

    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.capacity:
                # 移除最久未使用的项
                self.cache.popitem(last=False)

            self.cache[key] = value

    def clear(self):
        with self.lock:
            self.cache.clear()

def memoize_with_lru(cache_size=1000):
    def decorator(func):
        cache = LRUCache(cache_size)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 创建缓存键
            key = str(args) + str(sorted(kwargs.items()))

            # 检查缓存
            result = cache.get(key)
            if result is not None:
                return result

            # 计算结果并缓存
            result = func(*args, **kwargs)
            cache.put(key, result)

            return result

        return wrapper
    return decorator

class ModelCache:
    def __init__(self, max_models=5):
        self.max_models = max_models
        self.models = {}
        self.usage_count = {}
        self.last_used = {}
        self.lock = threading.Lock()

    def get_model(self, model_id, model_loader):
        with self.lock:
            if model_id in self.models:
                # 更新使用统计
                self.usage_count[model_id] += 1
                self.last_used[model_id] = time.time()
                return self.models[model_id]

            # 加载新模型
            model = model_loader()

            # 如果达到最大数量，移除最少使用的模型
            if len(self.models) >= self.max_models:
                self._remove_least_used()

            # 添加新模型
            self.models[model_id] = model
            self.usage_count[model_id] = 1
            self.last_used[model_id] = time.time()

            return model

    def _remove_least_used(self):
        # 基于使用次数和最后使用时间选择要移除的模型
        min_score = float('inf')
        model_to_remove = None

        for model_id in self.models:
            score = self.usage_count[model_id] / (time.time() - self.last_used[model_id])
            if score < min_score:
                min_score = score
                model_to_remove = model_id

        if model_to_remove:
            del self.models[model_to_remove]
            del self.usage_count[model_to_remove]
            del self.last_used[model_to_remove]
```