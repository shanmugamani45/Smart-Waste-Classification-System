import urllib.request
import json
from pathlib import Path

boundary = '----WebKitFormBoundaryTest7MA4'

def upload_img(file_path):
    with open(file_path, 'rb') as f:
        bytes_data = f.read()
    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
        f'Content-Type: image/jpeg\r\n\r\n'
    ).encode('utf-8') + bytes_data + f'\r\n--{boundary}--\r\n'.encode('utf-8')

    req = urllib.request.Request('http://127.0.0.1:8000/predict', data=body)
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode('utf-8'))

def post_feedback(pred_id, correct_class):
    fb_data = json.dumps({
        'prediction_id': pred_id,
        'user_feedback': 'incorrect',
        'corrected_class': correct_class
    }).encode('utf-8')
    req = urllib.request.Request('http://127.0.0.1:8000/feedback', data=fb_data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode('utf-8'))

# 1. Test paper image
paper_img = list(Path('dataset/test/paper').glob('*.jpg'))[0]
print(f"Testing image: {paper_img}")

pred1 = upload_img(paper_img)
p1_class = pred1['prediction']['class']
p1_conf = pred1['prediction']['confidence']
p1_warn = pred1['recommendation'].get('confidence_warning')
print(f"Step 1 - Initial Prediction: class='{p1_class}', conf={p1_conf:.2%}, warning='{p1_warn}'")

# 2. User submits feedback: correct class is 'paper'
fb_res = post_feedback(pred1['id'], 'paper')
print(f"Step 2 - Feedback Response: {fb_res['message']}")

# 3. User re-uploads the exact same image
pred2 = upload_img(paper_img)
p2_class = pred2['prediction']['class']
p2_conf = pred2['prediction']['confidence']
p2_warn = pred2['recommendation'].get('confidence_warning')
print(f"Step 3 - Re-upload Prediction: class='{p2_class}', conf={p2_conf:.2%}, warning='{p2_warn}'")

assert p2_class == 'paper', f"Expected paper, got {p2_class}"
assert 'Verified' in (p2_warn or ''), f"Expected verified warning, got {p2_warn}"
print("SUCCESS: Feedback loop works with instant memorization!")
