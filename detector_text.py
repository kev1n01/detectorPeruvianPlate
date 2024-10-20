import cv2
import easyocr
import matplotlib.pyplot as plt

image_path = './img/1.jpg'

img = cv2.imread(image_path)

reader = easyocr.Reader(['en', 'es'], gpu=False)

text_ = reader.readtext(image_path)

threshold = 0.45

for t in text_:
    print(t)
    bbox, text, score = t
    if (score > threshold):
        cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 2)
        cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()