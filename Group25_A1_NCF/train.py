import torch
from tqdm import tqdm


def train_model(model, train_loader, valid_loader, criterion, optimizer,
                n_epochs=20, early_stopping_patience=5, scheduler=None, device='mps'):
    train_losses = []
    valid_losses = []
    best_valid_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs} [Train]"):
            user_id = batch['user_id'].to(device)
            movie_id = batch['movie_id'].to(device)
            label = batch['label'].to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model(user_id, movie_id)
            loss = criterion(output, label)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(label)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        valid_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{n_epochs} [Valid]"):
                user_id = batch['user_id'].to(device)
                movie_id = batch['movie_id'].to(device)
                label = batch['label'].to(device)

                output = model(user_id, movie_id)
                loss = criterion(output, label)

                valid_loss += loss.item() * len(label)

        valid_loss /= len(valid_loader.dataset)
        valid_losses.append(valid_loss)

        # Print progress
        print(f"Epoch {epoch + 1}/{n_epochs}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Valid Loss: {valid_loss:.4f}")

        # Learning rate scheduler
        if scheduler:
            scheduler.step(valid_loss)

        # Early stopping
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"New best validation loss: {best_valid_loss:.4f}")
        else:
            patience_counter += 1
            print(f"EarlyStopping: {patience_counter}/{early_stopping_patience}")

            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, train_losses, valid_losses
