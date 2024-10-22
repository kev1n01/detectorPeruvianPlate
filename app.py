import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import time
from datetime import datetime, timedelta
import sqlite3
import re
import os
import torch

class PeruvianPlateDetector:
    def __init__(self):
        # Usar GPU si está disponible, de lo contrario usar CPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.plate_detector = YOLO('./weights/license_detector_medium.pt').to(device)  # Cargar modelo en GPU/CPU
        self.reader = easyocr.Reader(['es'], gpu=torch.cuda.is_available())  # Usar GPU si está disponible para EasyOCR
        self.db_connection = self.setup_database()  # Conexión a la base de datos
        self.last_processed_plates = {}
        self.min_detection_interval = 15  # Intervalo en segundos entre detecciones de la misma placa

    def setup_database(self):
        """Configura la base de datos para registrar vehículos y movimientos."""
        conn = sqlite3.connect('vehicle_registry.db')
        cursor = conn.cursor()

        # Tabla de vehículos registrados
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT UNIQUE,
                vehicle_type TEXT,
                first_seen DATETIME
            )
        ''')

        # Tabla de registros de movimiento
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicle_movements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT,
                movement_type TEXT,  -- 'entrada' o 'salida'
                timestamp DATETIME,
                image_path TEXT,
                confidence FLOAT,
                FOREIGN KEY (plate_number) REFERENCES vehicles(plate_number)
            )
        ''')

        conn.commit()
        return conn

    def validate_plate_format(self, plate_text):
        """Valida el formato de una placa peruana y determina el tipo de vehículo."""
        plate_text = plate_text.upper().strip()
        plate_text = re.sub(r'\s+', '', plate_text)  # Eliminar espacios

        # Eliminar prefijos 'PE' o 'PERU' si existen
        plate_text = re.sub(r'^(PE|PERU)[-]?', '', plate_text)
        
        # Definir los patrones válidos para placas peruanas
        patterns = [
            (r'^(\d{4})[-]?([A-Z]{2})$', 'moto'),      # Motocicleta/Mototaxi (1234-AB)
            (r'^([A-Z]{3})[-]?(\d{3})$', 'regular'),   # Vehículos regulares (ABC-123)
            (r'^E[\s-]?PA[-]?(\d{3})$', 'policia')     # Policía (E PA-123)
        ]

        for pattern, vehicle_type in patterns:
            match = re.match(pattern, plate_text)
            if match:
                if vehicle_type == 'moto':
                    formatted = f"{match.group(1)}-{match.group(2)}"
                elif vehicle_type == 'regular':
                    formatted = f"{match.group(1)}-{match.group(2)}"
                elif vehicle_type == 'policia':
                    formatted = f"E PA-{match.group(1)}"
                print("formato valida ",  formatted, "tipo ", vehicle_type)
                return formatted, vehicle_type

        return None, None

    def preprocess_plate(self, plate_img):
        """Preprocesa la imagen de la placa para mejorar la precisión del OCR."""
        # Redimensionar para mejorar el procesado y precisión del OCR
        min_width = 200
        if plate_img.shape[1] < min_width:
            aspect_ratio = plate_img.shape[0] / plate_img.shape[1]
            new_width = min_width
            new_height = int(min_width * aspect_ratio)
            plate_img = cv2.resize(plate_img, (new_width, new_height))

        # Convertir a escala de grises
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Mejorar contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Reducción de ruido con filtro bilateral (más efectivo para OCR)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Binarización adaptativa
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        
        return thresh

    def determine_movement_type(self, plate_number):
        """Determina si el movimiento es de 'entrada' o 'salida'."""
        cursor = self.db_connection.cursor()
        cursor.execute('''
            SELECT movement_type 
            FROM vehicle_movements 
            WHERE plate_number = ? 
            ORDER BY timestamp DESC 
            LIMIT 1
        ''', (plate_number,))

        last_movement = cursor.fetchone()

        if not last_movement:
            return 'entrada'
        return 'salida' if last_movement[0] == 'entrada' else 'entrada'

    def register_vehicle(self, plate_number, vehicle_type):
        """Registra un nuevo vehículo si no existe en la base de datos."""
        cursor = self.db_connection.cursor()
        cursor.execute('''
            INSERT OR IGNORE INTO vehicles (plate_number, vehicle_type, first_seen)
            VALUES (?, ?, ?)
        ''', (plate_number, vehicle_type, datetime.now()))
        self.db_connection.commit()

    def register_movement(self, plate_number, movement_type, image_path, confidence):
        """Registra un movimiento de entrada o salida."""
        cursor = self.db_connection.cursor()
        cursor.execute('''
            INSERT INTO vehicle_movements 
            (plate_number, movement_type, timestamp, image_path, confidence)
            VALUES (?, ?, ?, ?, ?)
        ''', (plate_number, movement_type, datetime.now(), image_path, confidence))
        self.db_connection.commit()

    def can_process_plate(self, plate_number):
        """Controla el intervalo entre detecciones de la misma placa."""
        current_time = datetime.now()
        if plate_number in self.last_processed_plates:
            last_time = self.last_processed_plates[plate_number]
            if (current_time - last_time) < timedelta(seconds=self.min_detection_interval):
                return False
        self.last_processed_plates[plate_number] = current_time
        return True

    def recognize_plate(self, plate_img):
        """Reconoce el texto de una placa de vehículo en una imagen."""
        processed_plate = self.preprocess_plate(plate_img)

        # Configurar parámetros de EasyOCR para detectar la placa
        results = self.reader.readtext(
            processed_plate,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ',
            batch_size=1,
            detail=1
        )

        if not results:
            return None, None, 0.0

        # Ordenar resultados por posición horizontal
        results.sort(key=lambda x: x[0][0][0])

        for t in results:
            bbox, text, score = t
            formatted_plate, vehicle_type = self.validate_plate_format(text)
            if formatted_plate:
                print("placa registrada", formatted_plate, "score ", score)
                return formatted_plate, vehicle_type, score
                
        return None, None, 0.0

    def process_video_feed(self):
        """Procesa el video en tiempo real para detectar y reconocer placas de vehículos."""
        cap = cv2.VideoCapture(1)  # Reemplazar con el índice de tu cámara o fuente de video
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.plate_detector(frame)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    if conf > 0.7:  # Umbral de confianza para detección
                        plate_img = frame[y1:y2, x1:x2]
                        plate_number, vehicle_type, ocr_conf = self.recognize_plate(plate_img)

                        if plate_number and ocr_conf > 0.6:  # Umbral de confianza para OCR
                            if self.can_process_plate(plate_number):
                                # Determinar tipo de movimiento
                                movement_type = self.determine_movement_type(plate_number)

                                # Guardar imagen
                                img_path = f'plates/{plate_number}_{movement_type}_{int(time.time())}.jpg'
                                
                                if not os.path.exists('plates'):
                                    os.makedirs('plates')
                                cv2.imwrite(img_path, plate_img)

                                # Registrar vehículo y movimiento
                                self.register_vehicle(plate_number, vehicle_type)
                                self.register_movement(plate_number, movement_type, img_path, ocr_conf)
                                
                                print("placa registrada", plate_number)
                                # Visualización en la pantalla
                                color = (0, 255, 0) if movement_type == 'entrada' else (0, 0, 255)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                cv2.putText(frame, 
                                           f'{plate_number} ({movement_type})',
                                           (x1, y1 - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX,
                                           0.9,
                                           color, 
                                           2)

            cv2.imshow('Placa detectada', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.db_connection.close()

if __name__ == "__main__":
    detector = PeruvianPlateDetector()
    detector.process_video_feed()
