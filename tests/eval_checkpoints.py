import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from src.models.model_factory import ModelFactory

def evaluate_checkpoint(ckpt_path):
    print(f"\nEvaluating: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model = ModelFactory.build_model('MobileNetV3', num_classes=7, pretrained=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    idx_to_class = ckpt.get('idx_to_class', {0: 'e_waste', 1: 'glass', 2: 'metal', 3: 'organic', 4: 'paper', 5: 'plastic', 6: 'textile'})
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    classes = ['e_waste', 'glass', 'metal', 'organic', 'paper', 'plastic', 'textile']
    print(f"{'Actual':<12} | {'Predicted':<12} | {'Confidence':<10} | {'Match'}")
    print("-" * 50)

    correct = 0
    total = 0
    for c in classes:
        p = Path(f'dataset/test/{c}')
        imgs = [f for f in p.glob('*.jpg') if not f.name.startswith('.')]
        if imgs:
            img = Image.open(imgs[0]).convert('RGB')
            t = transform(img).unsqueeze(0)
            with torch.no_grad():
                logits = model(t)
                probs = torch.softmax(logits, dim=-1)[0]
                pred_idx = torch.argmax(probs).item()
                conf = probs[pred_idx].item()
                pred_c = idx_to_class[pred_idx]
                match = (c == pred_c)
                if match:
                    correct += 1
                total += 1
                print(f"{c:<12} | {pred_c:<12} | {conf*100:>7.2f}%   | {'OK' if match else 'FAIL'}")

    print(f"Accuracy: {correct}/{total} ({correct/total*100:.2f}%)")

if __name__ == '__main__':
    evaluate_checkpoint('models/mobilenetv3_best.pt')
    evaluate_checkpoint('models/best_model.pt')
