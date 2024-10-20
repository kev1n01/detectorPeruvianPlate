import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import time
from datetime import datetime, timedelta
import sqlite3
import re
import os

class PeruvianPlateDetector:
    def __init__(self):
        self.plate_detector = YOLO('./weights/license_detector_medium.pt')  # Modelo de detección de placas
        self.reader = easyocr.Reader(['en'])  # inicializar lector de texto
        self.db_connection = self.setup_database()  # inicializar conexion a la base de datos
        self.last_processed_plates = {}
        self.min_detection_interval = 15  # intervalo en segundos entre detecciones

    def setup_database(self):
        """
        Establece la conexión con la base de datos y crea las tablas

        vehicles:
            id (INTEGER PRIMARY KEY AUTOINCREMENT): identificador único
            plate_number (TEXT UNIQUE): número de placa
            vehicle_type (TEXT): tipo de vehículo
            first_seen (DATETIME): fecha y hora de la primera detección

        vehicle_movements:
            id (INTEGER PRIMARY KEY AUTOINCREMENT): identificador único
            plate_number (TEXT): número de placa (llave foránea con vehicles)
            movement_type (TEXT): tipo de movimiento ('entrada' o 'salida')
            timestamp (DATETIME): fecha y hora del movimiento
            image_path (TEXT): ruta del archivo de la imagen
            confidence (FLOAT): confianza en la detección
        """

        if(os.path.exists('vehicle_registry.db')):
            os.create_file('vehicle_registry.db')

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
        
        # Tabla de registros de entrada/salida waaaaaa
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
        """
        Valida el formato de una placa de vehículo peruana y devuelve el formato
        estandarizado y el tipo de vehículo. Si no se ajusta a ninguno de los
        patrones, devuelve None, None.

        Formatos válidos:
        - Motocicleta/Mototaxi: 1234-AB
        - Vehículos regulares: ABC-123
        - Policía: E PA-123

        :param plate_text: Texto de la placa a validar
        :return: tuple (placa estandarizada, tipo de vehículo) o (None, None)
        """
        # limpieza inicial del texto
        plate_text = plate_text.upper().strip()
        plate_text = re.sub(r'\s+', '', plate_text)  # Eliminar espacios
        
        # elimina prefijos PE o PERU si existen
        plate_text = re.sub(r'^(PE|PERU)[-]?', '', plate_text)
        
        # Definiendo aatrones válidos
        patterns = [
            (r'^(\d{4})[-]?([A-Z]{2})$', 'moto'), # Caso 1: Motocicleta/Mototaxi (1234-AB)
            (r'^([A-Z0-9]{2,3})[-]?(\d{3})$', 'regular'), # Caso 2: Vehiculos regulares (ABC-123)
            (r'^E[\s-]?PA[-]?(\d{3})$', 'policia') # Caso 3: Policia (E PA-123)
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
                return formatted, vehicle_type
        
        return None, None

    def preprocess_plate(self, plate_img):
        """
        Preprocesa la imagen de entrada de una matrícula para mejorar el procesamiento y la precisión del OCR.

        Params:
        - plate_img: Imagen de entrada de la matrícula.

        Pasos:
        1. Cambie el tamaño de la imagen para mejorar el procesamiento y la precisión del OCR.
        2. Convierta la imagen a escala de grises.
        3. Mejore el contraste utilizando la ecualización de histograma adaptativa limitada por contraste (CLAHE).
        4. Aplique desenfoque gaussiano para reducir el ruido.
        5. Realice un umbral adaptativo para la binarización.
        6. Operaciones morfológicas para mejorar la calidad.

        Return:
        - Imagen preprocesada lista para OCR.
        """
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
        
        # Reducción de ruido
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        
        # Binarización adaptativa
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Operaciones morfológicas para mejorar la calidad
        kernel = np.ones((2,2), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        
        return thresh

    def determine_movement_type(self, plate_number):
        """Determina el tipo de movimiento (entrada o salida) para un vehículo
        registrando su número de placa.

        Busca el movimiento más reciente para el vehículo y si no existe, asume
        la entrada como primer movimiento. Si el movimiento más reciente es una
        entrada, el siguiente movimiento es una salida y viceversa.

        Args:
            plate_number (str): Número de placa del vehículo

        Returns:
            str: Tipo de movimiento ('entrada' o 'salida')
        """
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
        """Registra un nuevo vehículo si no existe."""
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
        """Reconoce el texto de una placa de vehículo en una imagen.

        Preprocesa la imagen de la placa, utiliza EasyOCR para detectar texto y
        ordena los resultados por posición horizontal. Luego, concatena los
        resultados y valida el formato de la placa. Si el formato es válido,
        calcula una confianza adicional basada en la longitud esperada y devuelve
        el número de placa, el tipo de vehículo y la confianza final.

        Args:
            plate_img (ndarray): Imagen de la placa de vehículo

        Returns:
            tuple: (número de placa, tipo de vehículo, confianza)
        """
        processed_plate = self.preprocess_plate(plate_img)
        
        # Configurar parámetros de EasyOCR para detectar PE/PERU
        results = self.reader.readtext(
            processed_plate,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ',
            batch_size=1,
            detail=1
        )
        
        if not results:
            return None, None, 0.0
        
        # orenar resultados por posición horizontal
        results.sort(key=lambda x: x[0][0][0])
        
        # Concatenar resultados
        text = ''.join(result[1] for result in results)
        confidence = np.mean([result[2] for result in results])
        
        # Validar formato de la placa
        plate_number, vehicle_type = self.validate_plate_format(text)
            
        if plate_number:
            # Calcular nueva confianza basada en la longitud esperada
            expected_length = len(plate_number)
            length_confidence = min(len(text) / (expected_length + 4), 1.0)  # +4 para considerar PE/PERU
            confidence = (confidence + length_confidence) / 2
        
        return plate_number, vehicle_type, confidence

    def test_plate_validation(self):
        """Función de prueba para validar diferentes formatos de placas."""
        test_cases = [
            "PE 1234-AB",
            "PERU 1234-AB",
            "PE ABC-123",
            "PERU ABC-123",
            "PE E PA-123",
            "PERU E PA-123",
            "1234-AB",
            "ABC-123",
            "E PA-123",
            "PE V2Q-424",
            "PERU V2Q-424",
            "V2Q-424"
        ]
        
        print("Prueba de validación de placas:")
        print("-" * 50)
        for plate in test_cases:
            formatted_plate, vehicle_type = self.validate_plate_format(plate)
            print(f"Original: {plate}")
            print(f"Formatted: {formatted_plate}")
            print(f"Type: {vehicle_type}")
            print("-" * 30)

    def process_video_feed(self):
        """
        Procesa el video en tiempo real para detectar y reconocer placas de vehículos.

        Mientras se ejecuta el bucle principal:
        - Captura un fotograma del video.
        - Utiliza el modelo de detección de placas para identificar regiones de interés.
        - Para cada región detectada:
            - Extrae la región de la placa.
            - Reconoce el texto de la placa y determina el tipo de vehículo.
            - Si la confianza en la detección y en el OCR supera ciertos umbrales:
                - Verifica si se puede procesar la placa nuevamente.
                - Determina el tipo de movimiento del vehículo.
                - Guarda la imagen de la placa.
                - Registra el vehículo y el movimiento en la base de datos.
                - Visualiza un cuadro alrededor de la placa con información de la placa y el movimiento.

        Al presionar 'q', se detiene la ejecución.

        """
        cap = cv2.VideoCapture(0)
        
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
                    
                    if conf > 0.5:  # Umbral de confianza para detección
                        plate_img = frame[y1:y2, x1:x2]
                        plate_number, vehicle_type, ocr_conf = self.recognize_plate(plate_img)
                        
                        if plate_number and ocr_conf > 0.7:  # Umbral de confianza para OCR
                            if self.can_process_plate(plate_number):
                                # Determinar tipo de movimiento
                                movement_type = self.determine_movement_type(plate_number)
                                
                                # Guardar imagen
                                img_path = f'plates/{plate_number}_{movement_type}_{int(time.time())}.jpg'
                                cv2.imwrite(img_path, plate_img)
                                
                                # Registrar vehículo y movimiento
                                self.register_vehicle(plate_number, vehicle_type)
                                self.register_movement(plate_number, movement_type, img_path, ocr_conf)
                                
                                # Visualización
                                color = (0, 255, 0) if movement_type == 'entrada' else (0, 0, 255)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                cv2.putText(frame, 
                                          f'{plate_number} ({movement_type})',
                                          (x1, y1-10),
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
    detector.test_plate_validation()
    detector.process_video_feed()