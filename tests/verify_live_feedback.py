import urllib.request
import urllib.parse
import json

def test():
    boundary = '----WebKitFormBoundaryTest7MA4'
    with open('dataset/test/plastic/verification_sample_1.jpg', 'rb') as f:
        file_bytes = f.read()

    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="sample.jpg"\r\n'
        f'Content-Type: image/jpeg\r\n\r\n'
    ).encode('utf-8') + file_bytes + f'\r\n--{boundary}--\r\n'.encode('utf-8')

    req = urllib.request.Request('http://127.0.0.1:8000/predict', data=body)
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    with urllib.request.urlopen(req) as resp:
        pred1 = json.loads(resp.read().decode('utf-8'))
    pred1_class = pred1['prediction']['class']
    pred1_conf = pred1['prediction']['confidence']
    pred1_id = pred1['id']
    print(f"1. Initial Predict: {pred1_class} (conf={pred1_conf:.2%}, id={pred1_id})")

    # Submit feedback changing to 'paper'
    fb_data = json.dumps({
        'prediction_id': pred1_id,
        'user_feedback': 'incorrect',
        'corrected_class': 'paper'
    }).encode('utf-8')
    req_fb = urllib.request.Request('http://127.0.0.1:8000/feedback', data=fb_data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req_fb) as resp:
        fb_res = json.loads(resp.read().decode('utf-8'))
    print(f"2. Feedback response: {fb_res['message']}")

    # Re-upload same image
    req2 = urllib.request.Request('http://127.0.0.1:8000/predict', data=body)
    req2.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    with urllib.request.urlopen(req2) as resp:
        pred2 = json.loads(resp.read().decode('utf-8'))
    pred2_class = pred2['prediction']['class']
    pred2_conf = pred2['prediction']['confidence']
    pred2_warning = pred2['recommendation'].get('confidence_warning')
    print(f"3. Re-upload: {pred2_class} (conf={pred2_conf:.2%}) warning={pred2_warning}")

    # Restore back to plastic
    fb_data2 = json.dumps({
        'prediction_id': pred2['id'],
        'user_feedback': 'correct',
        'corrected_class': 'plastic'
    }).encode('utf-8')
    req_fb2 = urllib.request.Request('http://127.0.0.1:8000/feedback', data=fb_data2, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req_fb2) as resp:
        json.loads(resp.read().decode('utf-8'))

    # Final verify
    req3 = urllib.request.Request('http://127.0.0.1:8000/predict', data=body)
    req3.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    with urllib.request.urlopen(req3) as resp:
        pred3 = json.loads(resp.read().decode('utf-8'))
    pred3_class = pred3['prediction']['class']
    pred3_conf = pred3['prediction']['confidence']
    print(f"4. Restored Predict: {pred3_class} (conf={pred3_conf:.2%})")

if __name__ == '__main__':
    test()
