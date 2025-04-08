# GANs_Assignment_4

---

# Training and Evaluating the LS-GAN, WGAN, and WGAN-GP on the Pneumonia MNIST Dataset

This project implements and evaluates three Generative Adversarial Network (GAN) variants — Least Squares GAN (LS-GAN), Wasserstein GAN (WGAN), and WGAN with Gradient Penalty (WGAN-GP) — on the Pneumonia MNIST dataset. The goal is to synthesize realistic chest X-ray images of pneumonia and assess the quality of each GAN's generation capability.

## 📁 Project Structure

```
.
├── data/
│   └── pneumonia_mnist/       # Pneumonia MNIST dataset (downloaded or extracted here)
├── models/
│   ├── ls_gan.py              # LS-GAN implementation
│   ├── wgan.py                # WGAN implementation
│   └── wgan_gp.py             # WGAN-GP implementation
├── utils/
│   ├── dataloader.py          # Data loading and preprocessing utilities
│   ├── trainer.py             # Training routines
│   └── evaluation.py          # Evaluation metrics (FID, Inception Score)
├── checkpoints/               # Directory for storing model checkpoints
├── samples/                   # Generated samples during training
├── train.py                   # Entry script to train models
├── evaluate.py                # Script to evaluate trained models
└── README.md
```

---

## 📦 Requirements

- Python 3.8+
- PyTorch
- torchvision
- numpy
- matplotlib
- tqdm
- scikit-learn
- Pillow
- [MedMNIST package](https://github.com/MedMNIST/MedMNIST) for easy access to the PneumoniaMNIST dataset

Install requirements:

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

We use the **PneumoniaMNIST** dataset from the MedMNIST collection.

- 28x28 grayscale X-ray images
- Three classes: normal, bacterial pneumonia, and viral pneumonia
- Used in binary classification: Normal (0) vs Pneumonia (1)

To download and prepare the dataset:

```python
from medmnist import PneumoniaMNIST
PneumoniaMNIST(download=True)
```

---

## 🚀 Training

Train LS-GAN, WGAN, or WGAN-GP by specifying the model in the command:

```bash
# Train LS-GAN
python train.py --model ls_gan

# Train WGAN
python train.py --model wgan

# Train WGAN-GP
python train.py --model wgan_gp
```

Optional arguments:
- `--epochs`: number of training epochs (default: 100)
- `--batch_size`: batch size (default: 64)
- `--lr`: learning rate (default: 0.0002)
- `--save_interval`: how often to save checkpoints and samples

---

## 📈 Evaluation

Evaluate FID or other metrics on generated images:

```bash
python evaluate.py --model wgan_gp --checkpoint checkpoints/wgan_gp_epoch_100.pth
```

Evaluations include:
- Frechet Inception Distance (FID)
- Inception Score (IS)
- Visual inspection of generated samples

---

## 📷 Sample Outputs

Generated examples during training are saved in the `samples/` directory and can be visualized using:

```bash
python visualize_samples.py --model wgan
```

---

## 🧠 GAN Variants Details

| Model    | Discriminator Loss | Generator Loss       | Notes                                         |
|----------|--------------------|----------------------|-----------------------------------------------|
| LS-GAN   | Least Squares Loss | Minimize L2 to "real"| Encourages more stable training than BCE     |
| WGAN     | Wasserstein Loss   | Minimize critic score| Requires weight clipping in critic            |
| WGAN-GP  | Wasserstein + GP   | + Gradient Penalty   | Improves on WGAN by addressing clipping issues|

---

## 📌 Notes

- Weight clipping is used in WGAN, while WGAN-GP uses gradient penalty.
- Models are trained on PneumoniaMNIST only. You can extend to other MedMNIST datasets.
- Evaluation metrics are approximate due to low image resolution.


## 📜 License

This project is licensed under the MIT License. See `LICENSE` for more details.

---
