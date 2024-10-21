# Importar las bibliotecas necesarias
import cv2
from ultralytics import YOLO
import easyocr
from datetime import datetime
import sqlite3

# Inicializar el modelo YOLO y EasyOCR
class PlateDetectionSystem:
    def __init__(self):
        self.plate_detector = YOLO('./weights/license_detector_medium.pt')  # Cargar tu modelo entrenado
        self.reader = easyocr.Reader(['en'])  # Inicializar EasyOCR
        self.db_connection = self.setup_database()

    def setup_database(self):
        conn = sqlite3.connect('vehicle_registry.db')
        cursor = conn.cursor()
        
        # Crear tabla para registro de vehículos
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicle_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT,
                entry_time DATETIME,
                image_path TEXT
            )
        ''')
        conn.commit()
        return conn

    def preprocess_plate(self, plate_img):
        # Convertir a escala de grises
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Aplicar umbralización adaptativa
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Reducir ruido
        processed = cv2.medianBlur(thresh, 3)
        return processed

    def recognize_plate(self, plate_img):
        # Preprocesar la imagen de la placa
        processed_plate = self.preprocess_plate(plate_img)
        
        # Realizar OCR
        results = self.reader.readtext(
            processed_plate,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ',
            batch_size=1,
            detail=1
        )
      
        # Extraer el texto de la placa
        plate_text = ''
        for detection in results:
            plate_text += detection[1]
        
        # Limpiar y formatear el texto de la placa
        plate_text = ''.join(e for e in plate_text if e.isalnum())
        return plate_text

    def save_entry(self, plate_number, image_path):
        cursor = self.db_connection.cursor()
        cursor.execute('''
            INSERT INTO vehicle_entries (plate_number, entry_time, image_path)
            VALUES (?, ?, ?)
        ''', (plate_number, datetime.now(), image_path))
        self.db_connection.commit()

    def process_video_feed(self):
        cap = cv2.VideoCapture(0)  # Usar 0 para la webcam o la dirección IP de tu cámara
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detectar placas usando YOLO
            results = self.plate_detector(frame)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Obtener coordenadas del bbox
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Extraer la región de la placa
                    plate_img = frame[y1:y2, x1:x2]
                    
                    # Reconocer texto de la placa
                    plate_text = self.recognize_plate(plate_img)
                    
                    if plate_text:
                        # Guardar imagen de la placa
                        # img_path = f'plates/{plate_text}_{int(time.time())}.jpg'
                        # cv2.imwrite(img_path, plate_img)
                        
                        # Registrar en la base de datos
                        # self.save_entry(plate_text, img_path)
                        
                        # Dibujar bbox y texto en el frame
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, plate_text, (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Mostrar el frame
            cv2.imshow('Plate Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.db_connection.close()

if __name__ == "__main__":
    system = PlateDetectionSystem()
    system.process_video_feed()