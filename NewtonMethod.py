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

    def fit(self, X_train, y_train, max_iter=20, epochs=10):
        """Train using Newton's Method (L-BFGS)"""
        
        # Store data
        if isinstance(X_train, np.ndarray):
            self.X_train_tensor = torch.from_numpy(X_train).float()
        else:
            self.X_train_tensor = X_train
            
        if isinstance(y_train, np.ndarray):
            self.y_train_tensor = torch.from_numpy(y_train).long()
        else:
            self.y_train_tensor = y_train

        # Setup Optimizer with MANUAL parameters
        self.optimizer = optim.LBFGS(self.parameters(), 
                                     lr=1, 
                                     max_iter=max_iter,
                                     history_size=10)

        print(f"--- Starting Training (L-BFGS, Max Iter: {max_iter}, Epochs: {epochs}) ---")

        # Track epoch losses
        epoch_losses = []
        
        # Training loop over epochs
        for epoch in range(epochs):
            # Optimization Step
            m = self._closure()
            self.optimizer.step(m)
            
            # Calculate and store loss after this epoch
            with torch.no_grad():
                logits = torch.matmul(self.X_train_tensor, self.W) + self.b
                loss = self.criterion(logits, self.y_train_tensor)
                epoch_losses.append(loss.item())
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")
                if loss < 1e-6:
                    break
        
        # Create loss chart
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs + 1), epoch_losses, marker='o', linewidth=2, markersize=6)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss Over Epochs (L-BFGS)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('training_loss_chart.png', dpi=150, bbox_inches='tight')
        print(f"\nLoss chart saved to 'training_loss_chart.png'")
        plt.show()
        
        # Cleanup
        with torch.no_grad():
            final_logits = torch.matmul(self.X_train_tensor, self.W) + self.b
            final_loss = self.criterion(final_logits, self.y_train_tensor)
            print(f"Training Complete. Final Loss: {final_loss.item():.6f}")
            
        del self.X_train_tensor
        del self.y_train_tensor

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