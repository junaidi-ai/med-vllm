"""Tests for medical entity recognition functionality."""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch

from medvllm.models.ner import MedicalNERModel

class TestMedicalEntityRecognition(unittest.TestCase):
    """Test cases for medical entity recognition."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = MedicalNERModel()
        self.sample_text = "Patient presents with chest pain and headache."
        self.expected_entities = [
            {"entity": "SYMPTOM", "word": "chest pain", "start": 24, "end": 34, "score": 0.95},
            {"entity": "SYMPTOM", "word": "headache", "start": 39, "end": 47, "score": 0.92}
        ]
    
    @patch('medvllm.models.ner.MedicalNERModel._load_model')
    def test_entity_recognition(self, mock_load):
        """Test basic entity recognition functionality."""
        # Mock the model's forward pass
        mock_model = MagicMock()
        mock_model.return_value = {"logits": torch.randn(1, 10, 5)}  # [batch, seq_len, num_labels]
        mock_load.return_value = mock_model
        
        # Test entity recognition
        entities = self.model.extract_entities(self.sample_text)
        
        # Verify basic structure
        self.assertIsInstance(entities, list)
        for entity in entities:
            self.assertIn("entity", entity)
            self.assertIn("word", entity)
            self.assertIn("score", entity)
    
    def test_confidence_threshold(self):
        """Test that entities below confidence threshold are filtered out."""
        with patch.object(self.model, 'predict', return_value=self.expected_entities):
            # Test with high threshold (should filter out all)
            entities = self.model.extract_entities(
                self.sample_text, 
                confidence_threshold=0.99
            )
            self.assertEqual(len(entities), 0)
            
            # Test with low threshold (should keep all)
            entities = self.model.extract_entities(
                self.sample_text,
                confidence_threshold=0.8
            )
            self.assertGreaterEqual(len(entities), 1)
    
    def test_batch_processing(self):
        """Test processing multiple texts in a batch."""
        texts = [
            "Patient reports chest pain",
            "No significant findings",
            "History of diabetes and hypertension"
        ]
        
        with patch.object(self.model, 'predict_batch') as mock_predict:
            mock_predict.return_value = [
                [{"entity": "SYMPTOM", "word": "chest pain", "score": 0.95}],
                [],
                [
                    {"entity": "CONDITION", "word": "diabetes", "score": 0.97},
                    {"entity": "CONDITION", "word": "hypertension", "score": 0.96}
                ]
            ]
            
            results = self.model.extract_entities_batch(texts)
            self.assertEqual(len(results), 3)
            self.assertEqual(len(results[0]), 1)  # One entity in first text
            self.assertEqual(len(results[1]), 0)  # No entities in second text
            self.assertEqual(len(results[2]), 2)  # Two entities in third text


if __name__ == "__main__":
    unittest.main()
