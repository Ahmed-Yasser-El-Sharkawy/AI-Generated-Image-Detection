# 🧠 AI-Generated Image Detection System

[**🚀 Live Demo on Hugging Face Spaces**](https://huggingface.co/spaces/Ahmed-El-Sharkawy/Detect_AI-generated_Image)

A robust deep learning-based system to distinguish between **real** and **AI-generated images** using convolutional neural networks. This project focuses on tackling image manipulation and synthetic content detection by identifying subtle pixel-level artifacts introduced by generative models.

---

## 📌 Key Features

* ✅ **Custom Dataset**: Real and fake images including multiple styles of AI-generated samples.
* ✅ **Deep Architectures**: Models built using MobileNet V2, MobileNet V3 (Small & Large), and ResNet-18/34.
* ✅ **Anti-Overfitting**: Integrated dropout, data augmentation, and early stopping techniques.
* ✅ **Model Performance**:

  * **MobileNet V3 Large**: 🏆 96.23% accuracy (best performing)
  * **ResNet-18**: 89.00% accuracy
  * **Ensemble Model**: 92.99% accuracy (via majority voting)
* ✅ **Real-Time Deployment**: Accessible web app via Hugging Face Spaces.
* ✅ **Optimized for Speed**: Lightweight architectures to support edge deployment.

---

## 🗂️ Project Structure

```
AI-Generated-Image-Detection/
├── MobileNet_v2/
│   ├── mobilenet.ipynb
│   ├── mobilenet-8-to-16-epoches.ipynb
│   ├── best_model6for_mobilnet_sdk_First_training_part_16Epoches.pth
│   ├── scores2_MobileNet.csv
│   └── ...
│
├── MobileNet_v3_large/
│   ├── MobileNet_V3_large.ipynb
│   ├── best_model3_mobilenetv3_large.pth
│   └── MobileNet_2V3ofData.csv
│
├── MobileNet_v3_small/
│   ├── MobileNet_V3_Small.ipynb
│   ├── best_model6_mobilenetv3_small.pth
│   ├── Accuracy_output.png, Loss_output.png
│   └── ...
│
├── RESNET18/
│   ├── resnet-18- var-2 after solved over fitting.ipynb
│   ├── RESNET-18-best_model9.pth
│   └── Overfitting analysis notebooks
│
├── RESNET34/
│   ├── detect-ai-generated-images-resnet34-firstquartdata.ipynb
│   └── ...
│
├── app-gradio.py                   # Gradio Web App Interface
├── test-results.ipynb              # Final metrics and confusion matrices
├── requirements.txt                # Environment setup
└── README.md
```

---

## 🧪 Models Used

| Model                 | Type            | Accuracy   |
| --------------------- | --------------- | ---------- |
| MobileNet V3 Large    | Lightweight CNN | **96.23%** |
| MobileNet V2          | CNN             | \~91.2%    |
| ResNet-18             | Deep CNN        | 89.00%     |
| **Ensemble (Voting)** | Combined        | 92.99%     |

---

## 🚀 Deployment

The model is deployed on [Hugging Face Spaces](https://huggingface.co/spaces/Ahmed-El-Sharkawy/Detect_AI-generated_Image) using **Gradio**, allowing real-time prediction for uploaded images.

### Run locally

```bash
git clone https://github.com/Ahmed-Yasser-El-Sharkawy/AI-Generated-Image-Detection.git
cd AI-Generated-Image-Detection
pip install -r requirements.txt
python app-gradio.py
```

---

## ⚙️ Training Details

* **Frameworks**: PyTorch, TensorFlow (optional), Gradio
* **Regularization**: Dropout, EarlyStopping
* **Augmentations**: Horizontal Flip, Random Rotation, Zoom, Noise
* **Optimization**: AdamW / SGD with cosine decay
* **Batch Size**: 32–64
* **Epochs**: 8–16 (best models fine-tuned)

---

## 📊 Evaluation Metrics

* Accuracy
* F1-Score
* Confusion Matrix
* ROC-AUC (in notebooks)

---

## 📌 Future Work

* 🧬 Extend to **Deepfake Video Detection**
* 📦 Convert to **ONNX or TensorRT** for mobile apps
* 🛡️ Integrate with browser plugins for real-time web image scanning
* 📉 Improve false positive/negative thresholds with adversarial examples

---

## 🤝 Contributions

Pull requests and collaboration ideas are welcome. You can:

* Submit improvements to model performance
* Add new AI-generated image sources
* Contribute to the Gradio UI/UX

---

## 📄 License

This project is licensed under the MIT License.
