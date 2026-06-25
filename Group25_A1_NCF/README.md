# 🎬 Neural Collaborative Filtering on MovieLens 1M

This project implements **Neural Collaborative Filtering (NCF)** for movie recommendation using the [MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/). It combines Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP) to learn user-item interactions.

---

## 📦 Features

- Downloads and preprocesses MovieLens 1M dataset
- Performs **negative sampling** to generate implicit feedback data
- Implements a **custom PyTorch Dataset**
- Defines the **NCF model architecture** (GMF + MLP fusion)
- Trains with early stopping and learning rate scheduling
- Evaluates using **Recall@10** and **NDCG@10**
- Plots and saves training curves
- Supports multiple model configurations
- Saves model checkpoints and results

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```


### 2. Run the model

```bash
python main.py
```

### 4. Create a plot of the results

```bash
python line_chart.py              #CREATE LINE CHART
python bar_chart.py               #CREATE BAR CHART
```

This will:
- Download and preprocess data
- Run several experiments with different embedding and MLP configurations
- Plot training/validation losses
- Evaluate each model and save results
- Show the Recall@10 and NDCG@10 results in line chart and bar chart

---

## ⚙️ Project Structure

```bash
.
├── data/                # Raw and processed data
├── results/             # Saved models, plots, and metrics
├── main.py              # Main entry point
├── model.py             # NCF model definition
├── preprocessing.py     # Dataset download, transform, split
├── train.py             # Training loop
├── evaluation.py        # Model evaluation functions
├── line_chart.py        # Line chart of final results
├── bar_chart.py         # bar chart of final results
├── README.md            # You're here!
```

---

## 🧪 Example Configurations

The following configurations are evaluated in `main()`:

```python
configurations = [
    (32, [64, 32, 16]),     # Small
    (64, [128, 64, 32]),    # Medium (default)
    (128, [256, 128, 64]),  # Large
    (64, [128, 64])         # Fewer layers
]
```

You can modify or add more in `main.py`.

---

## 📊 Metrics

Two evaluation metrics are used:

- **Recall@10**: Measures how many relevant items are in the top 10
- **NDCG@10**: Rewards correct ranking of relevant items

Results are stored in:
- `results/results.json`: All experiments
- `results/config_<config_id>.json`: Per-experiment details
- `results/loss_curves_<config_id>.png`: Training loss plots
- `results/architecture_comparison_line.png`: Line chart of final results
- `results/architecture_comparison_line.png`: bar chart of final results


---

## 🧼 Preprocessing

1. Downloads MovieLens 1M
2. Converts explicit ratings to binary feedback (rating >= 4.0 = positive)
3. Performs **negative sampling** for implicit learning
4. Splits data into train/validation/test
5. Returns `torch.utils.data.Dataset` objects

---

## 🧠 Training Details

- Optimizer: `Adam'
- Loss: `BCELoss`
- Early stopping with patience = 5
- Learning rate scheduler: `ReduceLROnPlateau`

---

## 💾 Saving

If `save_model=True`, the model and config are saved in:

```
results/ncf_model_<config_id>.pt
results/config_<config_id>.json
```

---

## 📈 Plotting

Training and validation loss curves are plotted and saved as PNGs:

```
results/loss_curves_<config_id>.png
```


Recall@10 and NDCG@10 results for different configurations are shown in line and bar charts as PNGs:

```
results/architecture_comparison_line.png
results/architecture_comparison_bar.png
```

---
