import urllib.request
import json
from pathlib import Path

boundary = '----WebKitFormBoundaryTest7MA4'
classes = ['e_waste', 'glass', 'metal', 'organic', 'paper', 'plastic', 'textile']

print(f"{'Actual':<12} | {'Predicted':<12} | {'Confidence':<10} | {'Status'}")
print("-" * 50)
correct = 0
total = 0

for c in classes:
    p = Path(f'dataset/test/{c}')
    imgs = [f for f in p.glob('*.jpg') if not f.name.startswith('.')]
    if imgs:
        img_path = imgs[0]
        with open(img_path, 'rb') as f:
            bytes_data = f.read()
        body = (
            f'--{boundary}\r\n'
            f'Content-Disposition: form-data; name="file"; filename="{img_path.name}"\r\n'
            f'Content-Type: image/jpeg\r\n\r\n'
        ).encode('utf-8') + bytes_data + f'\r\n--{boundary}--\r\n'.encode('utf-8')

        req = urllib.request.Request('http://127.0.0.1:8000/predict', data=body)
        req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        
        pred_c = data['prediction']['class']
        conf = data['prediction']['confidence']
        is_match = (c == pred_c)
        if is_match:
            correct += 1
        total += 1
        print(f"{c:<12} | {pred_c:<12} | {conf*100:>7.2f}%   | {'OK' if is_match else 'FAIL'}")

print("-" * 50)
print(f"Overall Test Accuracy: {correct}/{total} ({correct/total*100:.2f}%)")
