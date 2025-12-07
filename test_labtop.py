import unittest
import torch
import shutil
import os
from transformers import AutoTokenizer
from labtop_dataset import LabTOPDataset
from labtop_model import LabTOPModel

class TestLabTOP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = "test_output"
        os.makedirs(cls.test_dir, exist_ok=True)
        cls.tokenizer_name = "emilyalsentzer/Bio_ClinicalBERT"
        
    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
            
    def test_dataset_initialization(self):
        print("\nTesting Dataset Initialization...")
        dataset = LabTOPDataset(
            root="dummy_root",
            save_path=self.test_dir,
            tokenizer_name=self.tokenizer_name,
            stay_limit=10
        )
        self.assertIsNotNone(dataset.tokenizer)
        print("Dataset initialized successfully.")
        
    def test_model_initialization_and_forward(self):
        print("\nTesting Model Initialization and Forward Pass...")
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = LabTOPModel(tokenizer=tokenizer, d_model=32, n_heads=2, num_layers=2)
        
        # Create dummy input
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, len(tokenizer), (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len))
        
        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask)
        self.assertTrue(hasattr(outputs, "logits"))
        self.assertEqual(outputs.logits.shape, (batch_size, seq_len, len(tokenizer)))
        print("Model forward pass successful.")

if __name__ == "__main__":
    unittest.main()
