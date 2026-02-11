import torch
import torch.nn as nn
from models.heads import get_head
from losses.contrastive_loss import ContrastiveLoss
from trainer.clip_trainer import ClipTrainer
from data.ptbxl_dataset import PTBXL


def test_heads():
    """Test projection heads"""
    # Test linear head
    linear = get_head("linear", embedding_dim=768, projection_dim=128)
    x = torch.randn(4, 768)
    y = linear(x)
    assert y.shape == (4, 128), f"Expected shape (4, 128), got {y.shape}"
    print("✓ Linear head works")
    
    # Test MLP head
    mlp = get_head("mlp", embedding_dim=768, projection_dim=128)
    y = mlp(x)
    assert y.shape == (4, 128), f"Expected shape (4, 128), got {y.shape}"
    print("✓ MLP head works")


def test_contrastive_loss():
    """Test contrastive loss"""
    loss_fn = ContrastiveLoss(temperature=0.07)
    
    # Create normalized embeddings
    batch_size = 8
    embedding_dim = 128
    ts_emb = torch.randn(batch_size, embedding_dim)
    text_emb = torch.randn(batch_size, embedding_dim)
    
    # Normalize
    ts_emb = torch.nn.functional.normalize(ts_emb, p=2, dim=-1)
    text_emb = torch.nn.functional.normalize(text_emb, p=2, dim=-1)
    
    # Compute loss
    loss = loss_fn(ts_emb, text_emb)
    
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"
    print(f"✓ Contrastive loss works (loss: {loss.item():.4f})")


def test_dataset_structure():
    """Test dataset has required methods"""
    # Check methods exist
    assert hasattr(PTBXL, "__init__"), "Missing __init__"
    assert hasattr(PTBXL, "__len__"), "Missing __len__"
    assert hasattr(PTBXL, "__getitem__"), "Missing __getitem__"
    print("✓ Dataset structure is correct")


def test_trainer():
    """Test trainer initialization"""
    model = nn.Linear(128, 128)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = ContrastiveLoss()
    
    trainer = ClipTrainer(
        model=model,
        optimizer=optimizer,
        contrastive_loss=loss_fn,
        accelerator=None,
        max_epochs=10
    )
    
    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert trainer.max_epochs == 10
    print("✓ Trainer initialization works")


if __name__ == "__main__":
    print("\n=== Running Tests ===\n")
    
    test_heads()
    test_contrastive_loss()
    test_dataset_structure()
    test_trainer()
    
    print("\n✓ All Tests Passed!\n")
