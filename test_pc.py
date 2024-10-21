import cv2
import psutil
import GPUtil
import numpy as np
import time
from ultralytics import YOLO
import easyocr

class PerformanceTest:
    def __init__(self):
        # self.model = YOLO('yolo8n.pt')  # Usando modelo v8 pequeño para prueba
        self.model = YOLO('./weights/license_detector_medium.pt')  # Usando modelo yolo11 XDDDDD
        self.reader = easyocr.Reader(['en'])
        
    def check_system_resources(self):
        print("\n=== Verificación de Recursos del Sistema ===")
        
        # CPU Info
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_freq = psutil.cpu_freq()
        print(f"\nCPU:")
        print(f"Uso actual: {cpu_percent}%")
        print(f"Frecuencia actual: {cpu_freq.current:.2f}MHz")
        print(f"Núcleos físicos: {psutil.cpu_count(logical=False)}")
        print(f"Núcleos totales: {psutil.cpu_count()}")
        
        # RAM Info
        ram = psutil.virtual_memory()
        print(f"\nRAM:")
        print(f"Total: {ram.total / (1024**3):.2f}GB")
        print(f"Disponible: {ram.available / (1024**3):.2f}GB")
        print(f"Uso actual: {ram.percent}%")
        
        # GPU Info
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                print(f"\nGPU:")
                print(f"Nombre: {gpu.name}")
                print(f"Memoria total: {gpu.memoryTotal}MB")
                print(f"Memoria libre: {gpu.memoryFree}MB")
                print(f"Uso actual: {gpu.load * 100}%")
        except:
            print("\nNo se detectó GPU NVIDIA")

    def test_detection_speed(self, video_source=0, num_frames=100):
        print("\n=== Prueba de Velocidad de Detección ===")
        
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print("Error al abrir la cámara")
            return
        
        # Warmup
        print("\nCalentando el sistema...")
        for _ in range(10):
            ret, frame = cap.read()
            if ret:
                _ = self.model(frame)
        
        # Prueba de velocidad
        print("\nIniciando prueba de velocidad...")
        detection_times = []
        ocr_times = []
        total_times = []
        
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            start_time = time.time()
            
            # Detección YOLO
            det_start = time.time()
            results = self.model(frame)
            det_time = time.time() - det_start
            detection_times.append(det_time)
            
            # OCR (solo si se detectó algo)
            if len(results[0].boxes) > 0:
                ocr_start = time.time()
                # Tomar la primera detección para la prueba
                box = results[0].boxes[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_img = frame[y1:y2, x1:x2]
                _ = self.reader.readtext(plate_img)
                ocr_time = time.time() - ocr_start
                ocr_times.append(ocr_time)
            
            total_time = time.time() - start_time
            total_times.append(total_time)
            
            if (i + 1) % 10 == 0:
                print(f"Procesados {i + 1}/{num_frames} frames")
        
        cap.release()
        
        # Resultados
        print("\nResultados:")
        print(f"FPS promedio: {1.0/np.mean(total_times):.2f}")
        print(f"Tiempo promedio detección: {np.mean(detection_times)*1000:.2f}ms")
        if ocr_times:
            print(f"Tiempo promedio OCR: {np.mean(ocr_times)*1000:.2f}ms")
        print(f"Tiempo total promedio por frame: {np.mean(total_times)*1000:.2f}ms")
        
        # Evaluación del rendimiento
        fps = 1.0/np.mean(total_times)
        if fps >= 30:
            print("\nRendimiento: EXCELENTE ✅")
            print("El sistema puede funcionar en tiempo real sin problemas")
        elif fps >= 20:
            print("\nRendimiento: BUENO ✅")
            print("El sistema puede funcionar en tiempo real con pequeños retrasos")
        elif fps >= 10:
            print("\nRendimiento: ACEPTABLE ⚠️")
            print("El sistema puede funcionar pero con retrasos notables")
        else:
            print("\nRendimiento: INSUFICIENTE ❌")
            print("Se recomienda mejorar el hardware o optimizar el sistema")

if __name__ == "__main__":
    test = PerformanceTest()
    test.check_system_resources()
    test.test_detection_speed()