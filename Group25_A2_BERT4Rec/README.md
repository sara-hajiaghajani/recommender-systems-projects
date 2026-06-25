
# BERT4Rec Recommendation Model

## Overview

This project implements the BERT4Rec model, a sequential recommendation model based on the BERT (Bidirectional Encoder Representations from Transformers) architecture. It is designed to predict the next item a user might interact with based on their historical interaction sequence. The project uses the MovieLens 1M dataset for training and evaluation. The main functionalities include data preprocessing, model training with hyperparameter tuning, evaluation using Recall@10 and NDCG@10 metrics, and plotting of results.

##  Project Structure

```
.
├── data/
│   └── ml-1m/
│       └── ratings.dat       # MovieLens 1M dataset (needs to be placed here)
├── results/
│   └── results.json          # Stores evaluation metrics for different configurations
│   └── (plots will also be saved here, e.g., 6_Layers_comparison_plot.png)
├── evaluation.py             # Implements evaluation metrics (Recall@10, NDCG@10)
├── main.py                   # Main script to run the entire pipeline
├── model.py                  # Defines the BERT4Rec model architecture
├── plot.py                   # Utility for plotting training and evaluation results
├── preprocessing.py          # Handles data loading, preprocessing, and dataset creation
├── train.py                  # Contains the training loop, optimization, and learning rate scheduling
└── README.md                 # This file
```

##  Getting Started
### Install dependencies
The project is built using Python and relies on several libraries. You can install them using pip:

```bash
pip install -r requirements.txt
```
### Dataset
The project uses the **MovieLens 1M dataset**.
-   Download the dataset (e.g., from [GroupLens](https://grouplens.org/datasets/movielens/1m/)).
-   Create the directory structure `data/ml-1m/`.
-   Place the `ratings.dat` file into the `data/ml-1m/` directory.

The `main.py` script expects the data at `data/ml-1m/ratings.dat`.

To run the project (data preprocessing, training, and evaluation):

1.  **Ensure the dataset is in the correct location:** `data/ml-1m/ratings.dat`.
2.  **Execute the main script:**

    ```bash
    python main.py
    ```

The script will:
-   Load and preprocess the MovieLens 1M data.
-   Split the data into training, validation, and test sets.
-   Initialize the BERT4Rec model.
-   Train the model using the specified parameters, with early stopping based on validation NDCG@10.
-   Evaluate the best model on the test set.
-   Save the evaluation results to `results/results.json`.
-   The `plot.py` script can be used separately to visualize results from `results.json` files. For example, to plot results for configurations stored in `results/6 Layers/*.json` and save it as `results/6 Layers/6_Layers_comparison_plot.png`, you can run `python plot.py` (assuming the paths in `plot.py` are set up for this).

##  Configuration

The main parameters for data preprocessing, model architecture, and training can be configured directly in `main.py`:

### Data Preprocessing Parameters (in `main.py`):
-   `MIN_USER_INTERACTIONS`: Minimum number of interactions for a user to be included (default: 5).
-   `MAX_SEQ_LENGTH`: Maximum length of user interaction sequences (default: 50).
-   `RATING_THRESHOLD`: Minimum rating to consider an interaction positive (default: 4).
-   `TEST_SIZE`: Proportion of users for the test set (default: 0.15).
-   `VALID_SIZE`: Proportion of users for the validation set (default: 0.15).

### Model Parameters (in `main.py`):
-   `HIDDEN_SIZE`: Dimensionality of the hidden layers and embeddings (default: 256).
-   `NUM_ATTENTION_HEADS`: Number of attention heads in the Transformer encoder (default: 8).
-   `NUM_TRANSFORMER_LAYERS`: Number of layers in the Transformer encoder (default: 6).
-   `DROPOUT_PROB`: Dropout probability (default: 0.2).
-   `MASK_PROB`: Probability of masking an item in the input sequence during training (default: 0.5).

Note: We compared the different configurations (as detailed in the report) by manually adjusting them in main.py. The plots are organized by the number of layers. For each plot, we fixed the layer count and compared models with varying hidden sizes and masking ratios.

### Training Parameters (in `main.py`):
-   `BATCH_SIZE`: Batch size for training and evaluation (default: 64).
-   `LEARNING_RATE`: Initial learning rate for the AdamW optimizer (default: 3e-4).
-   `EPOCHS`: Maximum number of training epochs (default: 100).
-   `EARLY_STOPPING_PATIENCE`: Number of epochs to wait for improvement before stopping training (default: 15).
-   `OPTIMIZER_WEIGHT_DECAY`: Weight decay for the AdamW optimizer (default: 0.01).
-   `WARMUP_RATIO`: Ratio of total training steps for the learning rate warmup phase (default: 0.2).

### Special Tokens (defined across files, managed in `main.py`):
-   `PAD_TOKEN`: Used for padding sequences (ID: 0).
-   `MASK_TOKEN`: Used for the masked language modeling task (ID: `num_items + 1`, set dynamically during preprocessing in `main.py`).

##   Metric

-   **Console Logs:** Training progress (loss, learning rate per batch/epoch), validation metrics per epoch (Recall@10, NDCG@10), and final test set performance are printed to the console.
-   **Results File (`results/results.json`):** A JSON file storing a list of configurations and their corresponding performance metrics. Each entry includes:
    -   Hyperparameters like `hidden_size`, `masking_ratio`, `layer_size`.
    -   Lists of `train_losses`, `valid_ndcg`, `valid_recall` per epoch.
    -   Final `recall@10` and `ndcg@10` on the test set.
    -   Timestamp of the run.
-   **Plots:** If `plot.py` is run (or its functionality integrated), it can generate and save comparison plots of model performance (e.g., `results/6 Layers/6_Layers_comparison_plot.png` as an example path from the script).
