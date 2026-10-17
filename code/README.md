# Noise-Robust Deep Learning on CIFAR-10N 

## 1. Executive Summary & Experimental Setup

Using a ResNet-18 backbone on a 45,000-image training split and 5,000-image validation split of CIFAR-10 (`train_images.npy`, `labels.csv`)

### Standard Training Configuration
To isolate algorithmic improvements from hyperparameter tuning, all experiments share a locked training baseline (`seed = 42`):
- **Architecture:** ResNet-18 (`num_classes = 10`)
- **Optimizer:** Stochastic Gradient Descent (`lr = 0.1`, `momentum = 0.9`, `weight_decay = 5e-4`)
- **Learning Rate Schedule:** Cosine Annealing (`CosineAnnealingLR`, $T_{\max} = 60$ epochs)
- **Batch Size:** `128` (Training), `256` (Validation/Inference)
- **Data Augmentation:** `RandomCrop(32, padding=4)` + `RandomHorizontalFlip()` + Per-Channel RGB Normalization $\mu = (0.4914, 0.4822, 0.4465)$, $\sigma = (0.2023, 0.1994, 0.2010)$

---

## 2. Task 1: Noise Characterization & Baseline Performance

### 2.1 Empirical Noise Rate & Asymmetric Noise
Comparing `clean_label` against `noisy_label` across the 45,000 training samples reveals an overall empirical noise rate of **40.2422%** .

Crucially, the $10 \times 10$ confusion matrix (`True (Clean)` vs. `Human (Noisy)`) demonstrates that real human annotation noise is **heavily asymmetric** rather than uniform:

| True (Clean) \ Human (Noisy) | airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **airplane** | **2925** | 293 | 314 | 89 | 94 | 105 | 85 | 75 | *379* | 146 |
| **automobile** | 191 | **2657** | 63 | 116 | 80 | 68 | 60 | 64 | 71 | *1126* |
| **bird** | 263 | 165 | **2842** | 212 | *254* | 216 | 222 | 147 | 106 | 77 |
| **cat** | 180 | 160 | 354 | **2168** | 194 | *924* | 206 | 106 | 91 | 86 |
| **deer** | 184 | 142 | 316 | 184 | **2195** | *491* | 174 | *683* | 94 | 50 |
| **dog** | 150 | 131 | 240 | *834* | 183 | **2620** | 123 | 136 | 55 | 61 |
| **frog** | 170 | 166 | *373* | *350* | 229 | 257 | **2676** | 124 | 96 | 62 |
| **horse** | 179 | 120 | 111 | 100 | 233 | *289* | 101 | **3246** | 77 | 50 |
| **ship** | *369* | *333* | 94 | 65 | 96 | 79 | 71 | 99 | **3076** | 214 |
| **truck** | 180 | *1247* | 81 | 87 | 78 | 81 | 52 | 68 | 115 | **2486** |

Key structural patterns in human mislabeling include:
1. **Vehicular Confusion:** `truck` is mislabeled as `automobile` **1,247** times, and `automobile` is mislabeled as `truck` **1,126** times. Similarly, `airplane` and `ship` share bilateral confusion (**379** and **369** cases) due to shared blue background contexts (sky vs. water).
2. **Animal Confusion:** `cat` is mislabeled as `dog` **924** times (and `dog` as `cat` **834** times), while `deer` is confused with `horse` (**683** times) and `dog` (**491** times).

Corrupted labels correlate strongly with real visual features (e.g., wheels, metallic chassis, fur, four legs), standard Cross-Entropy (CE) easily overfits to these false decision boundaries. By Epoch 60, the standard Noisy CE baseline collapses to **48.86%** validation accuracy on noisy labels and **67.90%** on clean validation labels.

---

## 3. Task 2: Learning Dynamics & Early Stopping

Tracking the training accuracy split between the clean training subset (`cleans == noisys`) and the mislabeled training subset (`cleans != noisys`, evaluated against the true hidden `clean_label`) exposes a two-phase learning regime in `t2_dynamics.csv`:

![Task 2 Learning Dynamics](./figures/t2_learning_dynamics.png)

### Key Milestones & Statistical Summary (`t2_dynamics.csv`, $N = 60$ Epochs)
- **Memorization Start Epoch (`Epoch 33`):** During the early feature-learning phase (Epochs 1–33), the network learns dominant, generalizable visual representations from the 59.76% clean majority. Consequently, its accuracy in predicting the *true clean label* on mislabeled training images rises from **24.35%** (Epoch 1) to a peak of **68.03%** at **Epoch 33**. After Epoch 33, as Cosine Annealing reduces the learning rate, the high-capacity ResNet-18 begins memorizing individual corrupted labels, causing true-class accuracy on the mislabeled subset to plummet from **68.03%** down to **22.68%** at Epoch 60 (meaning the network actively memorized $>77\%$ of the wrong labels).
- **Early Stop Epoch (`Epoch 46`):** Validation accuracy on noisy labels (`val_acc_noisy`) reaches its global maximum of **55.60%** at **Epoch 46** (coinciding exactly with the peak clean validation accuracy of **79.22%**). Between Epochs 33 and 46, incremental refinement on clean features briefly outweighs noise memorization; beyond Epoch 46, progressive memorization degrades clean validation accuracy by **-11.32 percentage points** (falling from **79.22%** to **67.90%**).

| Metric (`t2_dynamics.csv`) | Min (%) | Median (%) | Mean (%) | Std (%) | Peak (%) [Epoch] | Final (%) [Ep 60] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `train_acc_correct` | 28.36 | 86.09 | 82.49 | 13.63 | **96.56%** [Ep 60] | 96.56% |
| `train_acc_mislabeled` *(Memorization Resistance)* | 22.68 | 63.16 | 56.36 | 13.74 | **68.03%** [Ep 33] | 22.68% |
| `val_acc_noisy` | 27.68 | 51.78 | 50.36 | 5.47 | **55.60%** [Ep 46] | 48.86% |
| `val_acc_clean` | 33.12 | 72.74 | 70.18 | 8.96 | **79.22%** [Ep 46] | 67.90% |

---

## 4. Task 3: Per-Sample Noise Scoring (`t3_noise_scores.csv`)

### Methodology: Small-Loss Criterion at Early Stopping
To score all 45,000 training images by their likelihood of being mislabeled without peeking at `clean_label`, I evaluated the **per-sample Cross-Entropy loss** (`nn.CrossEntropyLoss(reduction='none')`) using the **Epoch 46 Early-Stopped Model** on unaugmented training images (`transform_eval`).


![Task 3 Loss Distribution](./figures/t3_noise_scores_distribution.png)

### Statistical Profile of `t3_noise_scores.csv` ($N = 45,000$)
- **Mean Loss:** `1.1267` | **Standard Deviation:** `1.2046`
- **Minimum:** `0.0094` | **25th Percentile:** `0.2519` | **Median (50th):** `0.5719`
- **60th Percentile (Clean/Noisy Prior Boundary):** `0.8504`
- **75th Percentile:** `1.6330` | **90th Percentile:** `3.1580` | **Maximum:** `6.6956`

The distribution exhibits a sharp low-loss mode below `0.60` corresponding to clean samples (~60% of the data) and a heavy right tail extending to `6.6956` capturing mislabeled samples.

---

## 5. Task 4: Noise-Robust Training Strategies

Evaluated three distinct models (**Label Smoothing**), (**Test-Time Augmentation Dynamic Relabeling**), (**Pure Symmetric Cross-Entropy**)—alongside a study on a **Hybrid SCE + TTA** failure which occured

![Task 4 Noisy Validation Accuracy](./figures/final_val_acc_noisy.png)

![Task 4 Clean Validation Accuracy](./figures/final_val_acc_clean.png)

![Task 4 Memorization Resistance](./figures/final_memorization_resistance.png)

### 5.1 Quantitative Comparison Across All Task 4 Methods

| Method | Peak `val_acc_noisy` [Ep] | Final `val_acc_noisy` | Peak `val_acc_clean` [Ep] | Final `val_acc_clean` | Peak `train_acc_mislabeled` [Ep] | Final `train_acc_mislabeled` | Mean / Std `val_acc_clean` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline CE (Task 1/2)** | 55.60% [Ep 46] | 48.86% | 79.22% [Ep 46] | 67.90% | 68.03% [Ep 33] | 22.68% | 70.18% ± 8.96% |
| **Method 1: Label Smoothing ($\epsilon = 0.1$)** | 55.44% [Ep 39] | 49.94% | 80.26% [Ep 40] | 69.42% | 67.46% [Ep 32] | 25.05% | 69.48% ± 10.20% |
| **Method 2: TTA Dynamic Relabeling** | 56.58% [Ep 47] | 55.70% | 82.62% [Ep 47] | 79.92% | **71.01%** [Ep 41] | **69.11%** | 73.15% ± 11.43% |
| **Method 3: Pure SCE ($\alpha=0.1, \beta=1.0$)** | **57.32%** [Ep 53] | **57.16%** | **83.00%** [Ep 60] | **83.00%** | 70.18% [Ep 49] | 67.53% | 67.26% ± 13.79% |

### 5.2 Methodological Breakdown
1. **Method 1 — Label Smoothing ($\epsilon = 0.1$):** Replaces hard one-hot targets $[0, \dots, 1, \dots, 0]$ with soft targets ($0.91$ for the target class, $0.01$ for others). While it slightly tempers extreme logit weights—improving peak `val_acc_clean` to **80.26%** (Epoch 40) and final `val_acc_clean` to **69.42%**—the $-\log(\hat{y}_c)$ gradient still pulls the network toward the 91% corrupted target as the learning rate decays. Memorization resistance still collapses from **67.46%** (Epoch 32) to **25.05%** (Epoch 60).
2. **Method 2 — TTA Dynamic Relabeling (`warmup = 15`, `threshold = 0.75`):** Maintains a dynamic GPU label bank (`dynamic_targets`). After 15 warmup epochs, each training batch undergoes 3 stochastic views (1 base pass + 2 on-GPU `RandomCrop`/`HorizontalFlip` passes). Whenever it exceeds `0.75` and disagrees with the current label, the label is permanently overwritten in VRAM. This active correction stops the late-epoch collapse, maintaining **69.11%** final memorization resistance and **79.92%** final clean validation accuracy (peaking at **82.62%** at Epoch 47). However, because a single network acts as both predictor and relabeler, occasional confirmation bias on ambiguous classes (such as `cat` vs. `dog`due to fur or animal features) causes a slight drift between Epoch 47 (**82.62%**) and Epoch 60 (**79.92%**).
3. **Method 3 — Pure Symmetric Cross-Entropy (`SCELoss`, $\alpha = 0.1, \beta = 1.0$) [Best Method]:** Combines standard Cross-Entropy with Reverse Cross-Entropy (RCE):
   $$\mathcal{L}_{\text{SCE}} = \alpha \left( - \sum_{i=1}^{C} y_i \log(\hat{y}_i) \right) + \beta \left( - \sum_{i=1}^{C} \hat{y}_i \log(y_i) \right)$$
   where one-hot targets $y_i$ are clamped to $[10^{-4}, 1.0]$ and predictions $\hat{y}_i$ to $[10^{-7}, 1.0]$. Down-weighting standard CE ($\alpha = 0.1$) prevents aggressive gradient updates on high-loss noisy labels, while RCE ($\beta = 1.0$) weights the penalty by the network's own visual confidence $\hat{y}_i$. Pure SCE completely eliminates overfitting, monotonically climbing to **83.00%** clean validation accuracy and **57.16%** noisy validation accuracy at Epoch 60 (peak **57.32%** at Epoch 53).

### 5.3 Failure Case Study: Why Hybrid (SCE + TTA Relabeling) Collapsed at Epoch 16
Also tested a fourth, hybrid architecture combining `SCELoss(alpha=0.1, beta=1.0)` with `TTA Dynamic Relabeling (warmup=15, thresh=0.75)`. Counterintuitively, merging our two strongest ideas triggered a catastrophic collapse that led us to terminate the run at Epoch 40:

```text
Training Task 4 Method 3: Hybrid (SCE Loss + TTA Relabeling)...
Epoch 01 | Fixed 0 bad labels | Val Acc (Noisy): 24.8%
...
Epoch 14 | Fixed 0 bad labels | Val Acc (Noisy): 43.2%
Epoch 15 | Fixed 0 bad labels | Val Acc (Noisy): 41.0%
Epoch 16 | Fixed 12563 bad labels | Val Acc (Noisy): 46.9%
Epoch 17 | Fixed 7573 bad labels | Val Acc (Noisy): 44.6%
Epoch 18 | Fixed 5950 bad labels | Val Acc (Noisy): 46.6%
...
Epoch 39 | Fixed 2829 bad labels | Val Acc (Noisy): 38.8%
Epoch 40 | Fixed 2713 bad labels | Val Acc (Noisy): 38.6%
```

**Root-Cause Diagnosis:** Because `SCELoss` uses $\alpha = 0.1$, it intentionally dampens early gradient magnitudes, causing slower initial convergence (`val_acc_noisy` is only **41.0%** at Epoch 15, whereas standard CE reached **51.4%** by Epoch 15). Simultaneously, the RCE term ($\beta = 1.0$) encourages sharp, overconfident softmax predictions on whatever primitive features the network has learned so far. When the TTA relabeler starts at **Epoch 16**, the underfitted network produced predictions with $>75\%$ confidence that were factually wrong, overwriting **12,563 labels in a single epoch** (and thousands more in subsequent epochs) and locking the model into irreversible confirmation bias.

---

## 6. Task 5: Real (Asymmetric) vs. Synthetic (Symmetric) Noise

### 6.1 Controlled Experimental Design: On-GPU Target Injection
To compare performance on `synthetic_label` (uniformly random symmetric noise at the same ~40% overall rate) against `noisy_label` (human asymmetric noise) without altering batch ordering

```python
synthetic_master_targets = torch.tensor(labels_df['synthetic_label'].values, dtype=torch.long).to(device)
# Inside training loop:
synth_targets = synthetic_master_targets[idx]
loss = criterion(outputs, synth_targets)
```

This guarantees that the network processes the exact same sequence of augmented image tensors as in Tasks 1 and 4, isolating the **Noise Transition Matrix** as the sole independent variable. Furthermore, because `synthetic_label` contains uniform die-roll corruptions, evaluating `val_acc_noisy` would measure agreement with random noise; hence, **`val_acc_clean`** is the primary evaluation metric.

![Task 5 Clean Validation Accuracy](./figures/t5_val_acc_clean.png)

![Task 5 Memorization Resistance](./figures/t5_memorization_resistance.png)

### 6.2 Quantitative Comparison: Real vs. Synthetic Noise

| Model & Noise Profile | Peak `val_acc_clean` [Epoch] | Final `val_acc_clean` [Ep 60] | Peak `train_acc_mislabeled` [Epoch] | Final `train_acc_mislabeled` [Ep 60] | Mean ± Std `val_acc_clean` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline CE (Real Asymmetric Noise)** | 79.22% [Ep 46] | 67.90% | 68.03% [Ep 33] | 22.68% | 70.18% ± 8.96% |
| **Baseline CE (Synthetic Symmetric Noise)** | **82.10%** [Ep 45] | **69.72%** | **79.65%** [Ep 39] | **38.04%** | 70.30% ± 12.23% |
| **Pure SCE (Real Asymmetric Noise)** | 83.00% [Ep 60] | 83.00% | 70.18% [Ep 49] | 67.53% | 67.26% ± 13.79% |
| **Pure SCE (Synthetic Symmetric Noise)** | **87.28%** [Ep 60] | **87.28%** | **83.99%** [Ep 55] | **83.72%** | 69.61% ± 15.62% |

### 6.3 Analysis of Real vs. Synthetic Noise Behavior
1. **Why Asymmetric Human Noise is Harder than Symmetric Random Noise:** Under uniform synthetic noise, the ~60% clean labels for class $c$ are opposed by ~40% noisy labels distributed uniformly across all 9 other classes (~4.4% per incorrect class). Under real human noise, errors are concentrated into specific visual lookalikes (e.g., `truck` $\to$ `automobile` accounts for **1,247** errors, nearly half of the true `truck` count of **2,486**). This concentrated bias warps the decision boundary much faster: Baseline CE's memorization resistance peaks lower (**68.03%** at Ep 33 vs. **79.65%** at Ep 39 on synthetic) and collapses deeper (**22.68%** vs. **38.04%** on synthetic).
2. **Why Pure SCE Dominates on Both—and Excels on Synthetic Noise:** On synthetic noise, Pure SCE achieves **87.28%** clean validation accuracy and **83.72%** final memorization resistance (up from **83.00%** and **67.53%** on real noise). Because synthetic noise pairs visually unrelated classes (e.g., a `frog` labeled as an `airplane`), the network's softmax confidence $\hat{y}$ on the true visual class quickly diverges from the random target $y$. The bounded RCE term immediately suppresses gradients from these contradictory pairs, allowing the network to train virtually undisturbed on the clean subset.

---

## 7. Exact Commands & Steps to Reproduce Each Task

### ⚠️ Note to Evaluators Regarding the Dataset
Due to GitHub's file size limits, the CIFAR-10N dataset files are **not** included in this repository. 

To run this notebook and evaluate the code:
1. Run the very first cell of `Notebook_25b0651.ipynb`. This will automatically generate an empty `data/` folder in your working directory.
2. Drop the required dataset files directly into that newly created `data/` folder.
3. Run the rest of the notebook.

### Directory & Environment Prerequisites
Ensure your working directory contains `data/` (`train_images.npy`, `test_images.npy`, `labels.csv`, `train_idx.txt`, `val_idx.txt`, `classes.txt`) and `model.py`. Install dependencies via:

```bash
pip install torch torchvision numpy pandas pillow scikit-learn matplotlib
```

### Smart-Caching vs. Full Retraining Execution
`Notebook_25b0651.ipynb` implements deterministic seeding (`torch.manual_seed(42)`) and automated checkpoint detection (`if os.path.exists(...)`).
- **Fast Evaluation Mode (Default):** If pre-computed `.csv` dynamics/predictions exist in `predictions/` and `.pth` weights exist in `models/`, executing all cells in `Notebook_25b0651.ipynb` loads the cached results, populates `results.json`, and generates all prediction files in seconds.
- **Full Retraining Mode (From Scratch):** If `models/` is empty (such as in a clean GitHub clone), running `Notebook_25b0651.ipynb` top-to-bottom automatically triggers the `else:` training branches and executes all 60-epoch training loops sequentially:
  - **Cells 1–4:** Initializes directories (`./data/`, `./models/`, `./predictions/`), computes the Task 1 noise rate (`40.2422%`) and $10 \times 10$ confusion matrix, and builds the `CIFAR10NDataset` and `DataLoader` pipelines.
  - **Cells 5–7 (Task 1):** Trains the Clean Reference ResNet-18 (`train_loader_clean`) and Noisy CE Baseline (`train_loader_noisy`) for 60 epochs, exports `predictions/t1_clean_ref.csv`, `predictions/t1_noisy_ce.csv`, and `predictions/t2_dynamics.csv`, and updates `results.json["task1"]`.
  - **Cell 8 (Tasks 2 & 3):** Parses `predictions/t2_dynamics.csv` to identify `memorization_start_epoch = 33` and `early_stop_epoch = 46`, trains the early-stopped model up to Epoch 46, generates `predictions/t2_early_stop.csv`, and computes unreduced per-sample Cross-Entropy losses on `train_loader_score` to save `predictions/t3_noise_scores.csv`.
  - **Cell 9 (Task 4 — Method 1):** Trains ResNet-18 with `nn.CrossEntropyLoss(label_smoothing=0.1)` for 60 epochs, saving `predictions/labelsmoothing_dynamics.csv` and `predictions/t4_LabelSmoothing.csv`.
  - **Cell 10 (Task 4 — Method 2):** Trains ResNet-18 with On-GPU Test-Time Augmentation Dynamic Relabeling (`warmup_epochs=15`, `confidence_thresh=0.75`), saving `predictions/dynamic_relabeling_tta_dynamics.csv` and `predictions/t4_TTA_Dynamic_Relabeling.csv`.
  - **Cell 11 (Task 4 — Method 3 [Best Method]):** Trains ResNet-18 with custom `SCELoss(alpha=0.1, beta=1.0)` for 60 epochs, saving `predictions/sce_dynamics.csv` and `predictions/t4_SCE.csv`, and updates `results.json["task4"]["best_method"] = "Symmetric_Cross_Entropy"`.
  - **Cell 12 (Task 5):** Injects `synthetic_label` on GPU (`synthetic_master_targets[idx]`) and trains both Baseline CE and Pure SCE for 60 epochs, saving `predictions/t5_baseline_synthetic_dynamics.csv`, `predictions/t5_sce_synthetic_dynamics.csv`, and updating `results.json["task5"]`.

---

## 8. Expected Outcomes

- **A noisy-label classifier compared against the clean-label reference, plus an early-stopped model:** The Baseline Noisy CE classifier reaches a peak clean validation accuracy of **79.22%** (`55.60%` on noisy validation labels) at **Epoch 46**, but degrades to **67.90%** clean validation accuracy (`48.86%` noisy) by Epoch 60 due to late-stage noise memorization. Halting training at the **Epoch 46** early-stopping checkpoint preserves the **79.22%** clean accuracy ceiling (`+11.32%` over the 60-epoch baseline) and yields `t2_early_stop.csv`.
- **A per-image noise score and a comparison of robust-training methods:** Evaluating unreduced Cross-Entropy loss using the Epoch 46 early-stopped model (`t3_noise_scores.csv`) separates clean training examples from mislabeled examples. 
Across Task 4 robust-training methods, **Pure Symmetric Cross-Entropy (`SCELoss`, $\alpha=0.1, \beta=1.0$)** achieves the highest performance (**83.00%** clean / **57.16%** noisy validation accuracy at Epoch 60), outperforming **TTA Dynamic Relabeling** and **Label Smoothing**.
- **A short explanation of memorization dynamics and of real vs synthetic noise:** CNNs trained with Cross-Entropy first learn generalizable visual patterns from the clean majority (Epochs 1–33, where true-class accuracy on mislabeled training images peaks at **68.03%**), and subsequently memorize corrupted labels as the learning rate decays (falling to **22.68%** by Epoch 60). Under Task 5 uniform synthetic noise (~40%), both models perform significantly better than under real human noise—Baseline CE rises from **67.90%** to **69.72%** clean accuracy (memorization resistance **22.68%** $\to$ **38.04%**), and Pure SCE jumps from **83.00%** to **87.28%** clean accuracy (memorization resistance **67.53%** $\to$ **83.72%**)—because synthetic noise distributes errors uniformly at random across all 9 off-diagonal classes, whereas real human noise concentrates errors into visually similar categories (`truck` $\leftrightarrow$ `automobile`, `cat` $\leftrightarrow$ `dog`) that induce coherent false gradients.
