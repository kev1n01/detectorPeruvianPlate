import cv2
import numpy as np
from PIL import Image
import easyocr

capture = cv2.VideoCapture(0)

result_text = ''

while True:
    ret, frame = capture.read()

    if not ret:
        break

    # Obtener dimensiones del frame
    height, width, _ = frame.shape
    print(height, width)  # Ejemplo de salida: 480 640

    # Dibujar un rectángulo para el texto de la placa
    cv2.rectangle(frame, (450, 400), (650, 500), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, result_text, (460, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Definir área de interés para detectar la placa
    x1 = int(width / 3)
    x2 = int(x1 * 2)
    y1 = int(height / 3)
    y2 = int(y1 * 2)

    # Dibujar rectángulo de procesamiento
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Cortar el área de interés
    cut = frame[y1:y2, x1:x2]

    # Convertir a escala de grises
    gray = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)

    # Aplicar umbral para binarización
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    # for contour in contours:
    #     area = cv2.contourArea(contour)

    #     if area > 1000:
    #         x, y, w, h = cv2.boundingRect(contour)

    #         xpi = x + x1
    #         ypi = y + y1

    #         xpf = x + w + x1
    #         ypf = y + h + y1

    #         # Dibujar rectángulo alrededor de la placa
    #         cv2.rectangle(frame, (xpi, ypi), (xpf, ypf), (255, 255, 0), 2)
            
    #         # Extraer la posible placa
    #         placa = frame[ypi:ypf, xpi:xpf]

    #         # Convertir la posible placa a escala de grises
    #         placa_gray = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)

    #         # Usar easyOCR para reconocer el texto
    #         if placa_gray.shape[0] >= 36 and placa_gray.shape[1] >= 82:  # Verificar dimensiones mínimas de una placa
    #             reader = easyocr.Reader(['es'], gpu=False)

    #             # Leer el texto de la placa
    #             text_ = reader.readtext(
    #                 placa_gray,
    #                 allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-',
    #                 batch_size=1,
    #                 detail=1
    #             )

    #             # Procesar los resultados de easyOCR
    #             for t in text_:
    #                 bbox, text, score = t
    #                 if len(text) >= 7:  # Validar si cumple con el formato mínimo de una placa
    #                     print('Placa detectada:', text)
    #                     result_text = text
    #                 else:
    #                     print('No cumple con el formato de placa')

    #         break

    # Mostrar el frame procesado
    cv2.imshow('Detección de Placa', frame)

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar ventanas
capture.release()
cv2.destroyAllWindows()
