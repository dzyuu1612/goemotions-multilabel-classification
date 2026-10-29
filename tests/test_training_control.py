import unittest
from src.models.training_control import EarlyStopping


class TrainingControlTest(unittest.TestCase):
    def test_noise_does_not_reset_patience_but_real_improvement_does(self):
        stop=EarlyStopping(2,0.001)
        self.assertFalse(stop.update(0.4))
        self.assertFalse(stop.update(0.4005))
        self.assertFalse(stop.update(0.403))
        self.assertFalse(stop.update(0.402))
        self.assertTrue(stop.update(0.4031))

    def test_disabled_and_invalid(self):
        stop=EarlyStopping()
        self.assertFalse(stop.update(0.4))
        self.assertFalse(stop.update(0.3))
        with self.assertRaises(ValueError):stop.update(float('nan'))
        with self.assertRaises(ValueError):EarlyStopping(-1)
