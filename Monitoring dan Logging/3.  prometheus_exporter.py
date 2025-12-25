
import time
import random
from prometheus_client import start_http_server, Gauge, Counter, Summary

# --- KONFIGURASI PORT ---
PORT = 8000

# --- DEFINISI 10 METRIK (SYARAT ADVANCED) ---
# 1. Metrik Bisnis/Model
ACCURACY = Gauge('model_accuracy', 'Current Model Accuracy')
PRECISION = Gauge('model_precision', 'Current Model Precision')
RECALL = Gauge('model_recall', 'Current Model Recall')
F1_SCORE = Gauge('model_f1_score', 'Current Model F1 Score')
CONFIDENCE = Gauge('prediction_confidence', 'Average Prediction Confidence')

# 2. Metrik Sistem
CPU_USAGE = Gauge('system_cpu_usage', 'Simulated CPU Usage Percentage')
MEMORY_USAGE = Gauge('system_memory_usage', 'Simulated Memory Usage (MB)')
LATENCY = Gauge('request_latency_seconds', 'Inference Latency in Seconds')

# 3. Metrik Operasional
REQUEST_COUNT = Counter('inference_request_total', 'Total Inference Requests')
DATA_DRIFT = Gauge('data_drift_score', 'Data Drift Score (Lower is better)')

def generate_dummy_metrics():
    while True:
        # Update nilai dengan angka random agar grafik bergerak
        ACCURACY.set(random.uniform(0.85, 0.95))
        PRECISION.set(random.uniform(0.80, 0.90))
        RECALL.set(random.uniform(0.75, 0.88))
        F1_SCORE.set(random.uniform(0.82, 0.92))
        CONFIDENCE.set(random.uniform(0.70, 0.99))
        
        CPU_USAGE.set(random.uniform(20, 60))
        MEMORY_USAGE.set(random.uniform(512, 1024))
        LATENCY.set(random.uniform(0.1, 0.5))
        DATA_DRIFT.set(random.uniform(0.01, 0.05))
        
        REQUEST_COUNT.inc() # Tambah 1 setiap detik
        
        print("Mengirim data metrik ke port 8000...")
        time.sleep(2) # Update setiap 2 detik

if __name__ == '__main__':
    # Jalankan server metrics di port 8000
    start_http_server(PORT)
    print(f"✅ Exporter berjalan di http://localhost:{PORT}")
    generate_dummy_metrics()

