import unittest

from montecarlo.montecarlo import check_statistical_properties, simulate_gbm


class TestMontecarlo(unittest.TestCase):

    def test_simulation(self):
        """Simple test of a 3-commodity set of correlated variables"""
        S0 = (
            50,  # power
            40,  # fuels
            100,  # Volume
        )
        vols = [
            0.15,  # Power
            0.1,  # fuels
            0.1,  # volume
        ]

        rho12 = 0.5  # power- fuel
        rho13 = 0.1  # power - volume
        rho23 = -0.4  # fuel- volume

        corr = [[1, rho12, rho13],
                [rho12, 1, rho23],
                [rho13, rho23, 1]]

        T = 1
        S = simulate_gbm(S0, vols, corr, T=T, mu=0, n_steps=50, n_sim=1000, show_plt=False)
        delta = 0.02
        calculated_vols, calculated_corrs = check_statistical_properties(S, T)
        for orig_vol, calculated_vol in zip(vols, calculated_vols):
            self.assertAlmostEqual(orig_vol, calculated_vol, delta=delta)
        for orig_corr, calculated_corr in zip([rho12, rho13, rho23], calculated_corrs):
            self.assertAlmostEqual(orig_corr, calculated_corr, delta=delta)

    def test_simulation_2cdty_1sim(self):
        """Simple test of a 2-commodity set of correlated variables but just 1 scenario"""
        S0 = (
            50,  # power
            40,  # fuels
        )
        vols = [
            0.15,  # Power
            0.1,  # fuels
        ]

        rho12 = 0.5  # power- fuel

        corr = [[1, rho12],
                [rho12, 1]]

        T = 1
        S = simulate_gbm(S0, vols, corr, T=T, mu=0, n_steps=50, n_sim=1, show_plt=False)
        delta = 0.05    # With just 1 simulation errors are larger
        calculated_vols, calculated_corrs = check_statistical_properties(S, T)
        for orig_vol, calculated_vol in zip(vols, calculated_vols):
            self.assertAlmostEqual(orig_vol, calculated_vol, delta=delta)
        for orig_corr, calculated_corr in zip([rho12], calculated_corrs):
            self.assertAlmostEqual(orig_corr, calculated_corr, delta=delta)

    def test_simulation_1cdty(self):
        """Simple test of a 1-commodity simulation"""
        S0 = (
            50,  # power
        )
        vols = [
            0.15,  # Power
        ]

        corr = [[1]]
        corr = None

        T = 1
        S = simulate_gbm(S0, vols, corr, T=T, mu=0, n_steps=50, n_sim=1, show_plt=False)
        delta = 0.01    # With just 1 simulation errors are larger
        calculated_vols, calculated_corrs = check_statistical_properties(S, T)
        for orig_vol, calculated_vol in zip(vols, calculated_vols):
            self.assertAlmostEqual(orig_vol, calculated_vol, delta=delta)
        for orig_corr, calculated_corr in zip([], calculated_corrs):
            self.assertAlmostEqual(orig_corr, calculated_corr, delta=delta)
