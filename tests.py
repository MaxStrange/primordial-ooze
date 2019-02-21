"""
Tests for the PrimordialOoze library.
"""
import unittest
import primordialooze as po

class TestOoze(unittest.TestCase):
    """
    All the tests for the PrimordialOoze package.
    """
    def test_instantiate(self):
        """
        Merely test whether we can successfully instantiate a Simulation instance.
        """
        _ = po.Simulation()

if __name__ == '__main__':
    unittest.main()
