import unittest
import os
import shutil
from mnist_optimizer_comparison.train import main as train_main

class TestMNISTOptimizers(unittest.TestCase):
    def setUp(self):
        self.output_dir = 'optimizer_plots'
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def test_optimizers_performance(self):
        train_main()
        # Check if the output plots are generated
        optimizers = ['KieferWolfowitz', 'Adam', 'Adagrad', 'SGD']
        for optimizer in optimizers:
            self.assertTrue(os.path.exists(os.path.join(self.output_dir, f'{optimizer}_optimizer_loss.png')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'comparative_optimizer_loss.png')))

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

if __name__ == '__main__':
    unittest.main()