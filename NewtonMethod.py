import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class CustomLogisticRegression:
    # NOTE: We removed (nn.Module), so we don't need super()!
    def __init__(self, input_size, num_classes):
        # 1. Manually define Weights and Bias
        # requires_grad=True tells PyTorch to track gradients for these specific variables
        # W shape: (768, 7)
        self.W = torch.randn(input_size, num_classes, requires_grad=True)
        # b shape: (7,)
        self.b = torch.zeros(num_classes, requires_grad=True)
        
        # Use CrossEntropyLoss (works the same way)
        self.criterion = nn.CrossEntropyLoss()
        
    def parameters(self):
        """
        We must manually tell the optimizer which variables to update
        because we aren't using nn.Module anymore.
        """
        return [self.W, self.b]

    def forward(self, x):
        # Ensure input is a tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # 2. Manual Linear Equation (The Math)
        # z = x * W + b
        z = torch.matmul(x, self.W) + self.b
        
        # 3. Softmax
        probs = torch.softmax(z, dim=1)
        return probs

    def predict(self, x):
        """Inference helper"""
        # We don't have .eval(), so we just use torch.no_grad()
        with torch.no_grad():
            probs = self.forward(x)
            return probs.numpy()

    # --- Boilerplate to allow calling model(x) directly ---
    def __call__(self, x):
        return self.forward(x)
    
    # --- Placeholders to prevent Pipeline crash ---
    def eval(self): pass
    def train(self): pass
    
    def save_weights(self, filepath):
        """Save the trained weights and bias to a file"""
        torch.save({'W': self.W, 'b': self.b}, filepath)
        print(f"Model weights saved to {filepath}")

    def load_weights(self, filepath):
        """Load weights from a file"""
        checkpoint = torch.load(filepath)
        with torch.no_grad():
            self.W.copy_(checkpoint['W'])
            self.b.copy_(checkpoint['b'])
        print(f"Model weights loaded from {filepath}")

    def load_state_dict(self, state_dict): 
        # Compatibility for the pipeline script if it calls this
        print("Using manual load_weights instead.")

    def _closure(self):
        """L-BFGS closure function"""
        self.optimizer.zero_grad()
        
        # Re-calculate logic manually using self.W and self.b
        logits = torch.matmul(self.X_train_tensor, self.W) + self.b
        loss = self.criterion(logits, self.y_train_tensor)
        
        loss.backward()
        return loss

    def fit(self, X_train, y_train, X_val=None, y_val=None, max_iter=20, epochs=10):
        """
        Train using Newton's Method (L-BFGS)
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            max_iter: Maximum iterations for L-BFGS
            epochs: Number of training epochs
        
        Returns:
            train_losses: List of training losses per epoch
            val_losses: List of validation losses per epoch (if validation data provided)
        """
        
        # Store training data
        if isinstance(X_train, np.ndarray):
            self.X_train_tensor = torch.from_numpy(X_train).float()
        else:
            self.X_train_tensor = X_train
            
        if isinstance(y_train, np.ndarray):
            self.y_train_tensor = torch.from_numpy(y_train).long()
        else:
            self.y_train_tensor = y_train
        
        # Store validation data if provided
        if X_val is not None:
            if isinstance(X_val, np.ndarray):
                self.X_val_tensor = torch.from_numpy(X_val).float()
            else:
                self.X_val_tensor = X_val
                
            if isinstance(y_val, np.ndarray):
                self.y_val_tensor = torch.from_numpy(y_val).long()
            else:
                self.y_val_tensor = y_val
        else:
            self.X_val_tensor = None
            self.y_val_tensor = None

        # Setup Optimizer with MANUAL parameters
        self.optimizer = optim.LBFGS(self.parameters(), 
                                     lr=1, 
                                     max_iter=max_iter,
                                     history_size=10)

        print(f"--- Starting Training (L-BFGS, Max Iter: {max_iter}, Epochs: {epochs}) ---")
        if X_val is not None:
            print(f"Validation set provided: {len(X_val)} samples")

        # Track epoch losses
        train_losses = []
        val_losses = []
        
        # Training loop over epochs
        for epoch in range(epochs):
            # Optimization Step
            # L-BFGS step() expects a callable closure function, not the result
            self.optimizer.step(self._closure)
            
            # Calculate and store training loss after this epoch
            with torch.no_grad():
                logits = torch.matmul(self.X_train_tensor, self.W) + self.b
                train_loss = self.criterion(logits, self.y_train_tensor)
                train_losses.append(train_loss.item())
                
                # Calculate validation loss if validation data provided
                val_loss_val = None
                if self.X_val_tensor is not None:
                    val_logits = torch.matmul(self.X_val_tensor, self.W) + self.b
                    val_loss_val = self.criterion(val_logits, self.y_val_tensor)
                    val_losses.append(val_loss_val.item())
                    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss.item():.6f}, Val Loss: {val_loss_val.item():.6f}")
                else:
                    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss.item():.6f}")
                
                if train_loss < 1e-6:
                    break
        
        # Store losses as instance variables for access from Training.py
        self.train_losses = train_losses
        self.val_losses = val_losses if val_losses else None
        
        # Cleanup
        with torch.no_grad():
            final_logits = torch.matmul(self.X_train_tensor, self.W) + self.b
            final_loss = self.criterion(final_logits, self.y_train_tensor)
            print(f"\nTraining Complete. Final Train Loss: {final_loss.item():.6f}")
            if self.X_val_tensor is not None:
                final_val_logits = torch.matmul(self.X_val_tensor, self.W) + self.b
                final_val_loss = self.criterion(final_val_logits, self.y_val_tensor)
                print(f"Final Validation Loss: {final_val_loss.item():.6f}")
            
        del self.X_train_tensor
        del self.y_train_tensor
        if self.X_val_tensor is not None:
            del self.X_val_tensor
            del self.y_val_tensor
        
        return train_losses, val_losses if val_losses else None

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    X_dummy = np.random.randn(100, 768).astype(np.float32)
    y_dummy = np.random.randint(0, 7, size=(100,))

    # 1. Initialize (No super init happens here!)
    model = CustomLogisticRegression(input_size=768, num_classes=7)
    
    # 2. Fit
    model.fit(X_dummy, y_dummy, max_iter=50, epochs=10)
    
    # 3. Save
    model.save_weights('dummy_model.pth')