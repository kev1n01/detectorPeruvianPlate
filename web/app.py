# app.py
from flask import Flask, render_template, Response, jsonify
import cv2
import sqlite3
from datetime import datetime
import threading
import queue
import time

app = Flask(__name__)

# Cola para comunicación entre hilos
detections_queue = queue.Queue()

# Configuración de la base de datos
def init_db():
    conn = sqlite3.connect('vehicles.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS vehicle_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            type TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def add_record(plate_number, record_type):
    conn = sqlite3.connect('vehicles.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO vehicle_records (plate_number, timestamp, type)
        VALUES (?, ?, ?)
    ''', (plate_number, datetime.now(), record_type))
    conn.commit()
    conn.close()

def get_latest_records():
    conn = sqlite3.connect('vehicle_registry.db')
    c = conn.cursor()
    records = c.execute('''
        SELECT plate_number, timestamp, movement_type 
        FROM vehicle_movements 
        ORDER BY timestamp DESC 
        LIMIT 10
    ''').fetchall()
    conn.close()
    return records

def detect_plate(frame):
    # Aquí va tu código actual de detección de placas
    # Este es solo un ejemplo, reemplázalo con tu implementación
    # Retorna la placa detectada o None si no se detecta ninguna
    return None

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Detectar placa
        plate = detect_plate(frame)
        if plate:
            # Agregar a la cola para procesamiento
            detections_queue.put(plate)
        
        # Convertir frame para streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_records')
def get_records():
    records = get_latest_records()
    return jsonify([{
        'plate': r[0],
        'timestamp': r[1],
        'type': r[2]
    } for r in records])

def process_detections():
    while True:
        if not detections_queue.empty():
            plate = detections_queue.get()
            # Determinar si es entrada o salida (implementar tu lógica)
            record_type = "entrada"  # o "salida"
            add_record(plate, record_type)
        time.sleep(0.1)

if __name__ == '__main__':
    init_db()
    # Iniciar hilo para procesar detecciones
    detection_thread = threading.Thread(target=process_detections, daemon=True)
    detection_thread.start()
    app.run(debug=True)