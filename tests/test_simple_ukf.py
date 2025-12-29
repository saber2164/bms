import unittest
import numpy as np
import sys
import os

# Add project root to path to import scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.simple_ukf import SimplifiedDualUKF

class TestSimplifiedDualUKF(unittest.TestCase):
    def setUp(self):
        self.dt = 1.0
        self.q_max = 2.0
        self.r0 = 0.05
        self.ukf = SimplifiedDualUKF(self.dt, self.q_max, self.r0)

    def test_initialization(self):
        """Test if the filter initializes with correct parameters"""
        self.assertEqual(self.ukf.dt, self.dt)
        self.assertEqual(self.ukf.theta[0], self.q_max)
        self.assertEqual(self.ukf.theta[1], self.r0)
        # Check initial state [SoC, U_d]
        self.assertEqual(self.ukf.x[0], 0.9) # Default init
        self.assertEqual(self.ukf.x[1], 0.0)

    def test_linear_ocv_monotonicity(self):
        """Test that the linear OCV model is strictly monotonic"""
        socs = np.linspace(0, 1, 100)
        ocvs = [self.ukf._linear_ocv(s) for s in socs]
        
        # Check if sorted (monotonic increasing)
        self.assertTrue(np.all(np.diff(ocvs) > 0), "OCV curve should be strictly increasing")
        
        # Check endpoints
        self.assertAlmostEqual(self.ukf._linear_ocv(0.0), 3.0)
        self.assertAlmostEqual(self.ukf._linear_ocv(1.0), 4.2)

    def test_prediction_step(self):
        """Test a single prediction step"""
        # Initial state
        initial_soc = 0.9
        self.ukf.x[0] = initial_soc
        
        # Simulate discharge: 1A current
        current = 1.0
        temp = 25.0
        
        # Expected voltage approx: OCV(0.9) - I*R0 = (1.2*0.9 + 3.0) - 1.0*0.05 = 4.08 - 0.05 = 4.03V
        # The filter should adjust SoC slightly based on this measurement
        measured_voltage = 4.03 
        
        soc, q_max, r0 = self.ukf.step(measured_voltage, current, temp)
        
        # SoC should decrease slightly due to discharge (Coulomb counting)
        # dSoC = -I*dt/Q = -1*1 / (2*3600) = -1.38e-4
        expected_coulomb_soc = initial_soc - (current * self.dt / (self.q_max * 3600))
        
        # The UKF estimate should be close to coulomb counting for one step with good measurement
        self.assertAlmostEqual(soc, expected_coulomb_soc, places=3)
        
        # Parameters should remain relatively stable in one step
        self.assertAlmostEqual(q_max, self.q_max, places=1)
        self.assertAlmostEqual(r0, self.r0, places=2)

    def test_soc_bounds(self):
        """Test that SoC stays within [0, 1]"""
        self.ukf.x[0] = 1.0
        # Simulate charging at full capacity (should stay at 1.0)
        # Note: The current implementation clips at the end of step
        self.ukf.step(4.5, -1.0, 25.0) # Charging
        self.assertLessEqual(self.ukf.x[0], 1.0)
        
        self.ukf.x[0] = 0.0
        # Simulate discharging at empty (should stay at 0.0)
        self.ukf.step(2.5, 1.0, 25.0) # Discharging
        self.assertGreaterEqual(self.ukf.x[0], 0.0)

if __name__ == '__main__':
    unittest.main()
