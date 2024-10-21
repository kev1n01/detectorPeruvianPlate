import cv2
import easyocr
import matplotlib.pyplot as plt

image_path = './img/plates.jpg'

img = cv2.imread(image_path)

reader = easyocr.Reader(['es'], gpu=False, )

text_ = reader.readtext(
        image_path,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ',
        batch_size=1,
        detail=1
    )

threshold = 0.45

for t in text_:
    bbox, text, score = t
    print(len(text))
    if(len(text) < 7 and  "-" not in text): # no cumple con el formato de placa
        print('no cumple con el formato de placa')
    else:
        text_r = text[:-1]
        print('bbox:',  bbox, 'text:' , text, 'score:' , score)
        print('text format:',  text_r)
        if (score > threshold):
            cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 2)
            cv2.putText(img, text_r, bbox[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()