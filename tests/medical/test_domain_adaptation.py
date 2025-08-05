"""Tests for medical domain adaptation functionality."""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch

from medvllm.models.domain_adaptation import DomainAdapter


class TestDomainAdaptation(unittest.TestCase):
    """Test cases for domain adaptation in medical models."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.source_domain = "general_medical"
        self.target_domain = "cardiology"
        self.feature_dim = 768  # Standard BERT hidden size
        self.batch_size = 4
        self.seq_length = 128
        
        # Create a mock model with an adapter
        self.model = MagicMock()
        self.adapter = DomainAdapter(
            model=self.model,
            source_domain=self.source_domain,
            target_domain=self.target_domain,
            feature_dim=self.feature_dim
        )
        
        # Create sample input data
        self.sample_input = {
            "input_ids": torch.randint(0, 30000, (self.batch_size, self.seq_length)),
            "attention_mask": torch.ones((self.batch_size, self.seq_length), dtype=torch.long),
            "domain": self.target_domain
        }
    
    def test_initialization(self):
        """Test that the domain adapter initializes correctly."""
        self.assertEqual(self.adapter.source_domain, self.source_domain)
        self.assertEqual(self.adapter.target_domain, self.target_domain)
        self.assertTrue(hasattr(self.adapter, 'domain_classifier'))
        self.assertTrue(hasattr(self.adapter, 'domain_embeddings'))
    
    def test_forward_pass(self):
        """Test the forward pass with domain adaptation."""
        # Mock the model's forward pass
        mock_output = {
            "last_hidden_state": torch.randn(self.batch_size, self.seq_length, self.feature_dim),
            "pooler_output": torch.randn(self.batch_size, self.feature_dim)
        }
        self.model.return_value = mock_output
        
        # Test forward pass with domain adaptation
        output = self.adapter(**self.sample_input)
        
        # Check that the output contains the expected keys
        self.assertIn("last_hidden_state", output)
        self.assertIn("pooler_output", output)
        self.assertIn("domain_logits", output)
        self.assertIn("domain_loss", output)
        
        # Check output shapes
        self.assertEqual(output["last_hidden_state"].shape, 
                        (self.batch_size, self.seq_length, self.feature_dim))
        self.assertEqual(output["pooler_output"].shape, 
                        (self.batch_size, self.feature_dim))
        self.assertEqual(output["domain_logits"].shape, 
                        (self.batch_size, 2))  # Binary classification
    
    def test_domain_adaptation_loss(self):
        """Test the domain adaptation loss calculation."""
        # Create mock logits and domain labels
        batch_size = 8
        logits = torch.randn(batch_size, 2)  # [batch_size, num_domains]
        
        # Test with all source domain samples
        domain_labels = torch.zeros(batch_size, dtype=torch.long)
        loss_source = self.adapter.domain_adaptation_loss(logits, domain_labels)
        
        # Test with all target domain samples
        domain_labels = torch.ones(batch_size, dtype=torch.long)
        loss_target = self.adapter.domain_adaptation_loss(logits, domain_labels)
        
        # Test with mixed domain samples
        domain_labels = torch.randint(0, 2, (batch_size,), dtype=torch.long)
        loss_mixed = self.adapter.domain_adaptation_loss(logits, domain_labels)
        
        # Check that losses are valid
        self.assertGreaterEqual(loss_source.item(), 0)
        self.assertGreaterEqual(loss_target.item(), 0)
        self.assertGreaterEqual(loss_mixed.item(), 0)
    
    def test_domain_classifier(self):
        """Test the domain classifier outputs."""
        # Create test features
        features = torch.randn(self.batch_size, self.feature_dim)
        
        # Get domain logits
        logits = self.adapter.domain_classifier(features)
        
        # Check output shape
        self.assertEqual(logits.shape, (self.batch_size, 2))  # Binary classification
    
    @patch('medvllm.models.domain_adaptation.DomainAdapter.domain_adaptation_loss')
    def test_training_step(self, mock_loss):
        """Test the training step with domain adaptation."""
        # Mock the loss
        mock_loss.return_value = torch.tensor(0.5)
        
        # Create a batch with both source and target domain samples
        batch = {
            "input_ids": torch.randint(0, 30000, (self.batch_size, self.seq_length)),
            "attention_mask": torch.ones((self.batch_size, self.seq_length), dtype=torch.long),
            "domain": torch.cat([
                torch.zeros(self.batch_size // 2),  # Source domain
                torch.ones(self.batch_size // 2)    # Target domain
            ]).long()
        }
        
        # Mock model outputs
        self.model.return_value = {
            "last_hidden_state": torch.randn(self.batch_size, self.seq_length, self.feature_dim),
            "pooler_output": torch.randn(self.batch_size, self.feature_dim)
        }
        
        # Run training step
        loss = self.adapter.training_step(batch, batch_idx=0)
        
        # Check that the loss is computed
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.item(), 0.5)  # Should match our mock
        
        # Check that the domain adaptation loss was called
        mock_loss.assert_called_once()


if __name__ == "__main__":
    unittest.main()
