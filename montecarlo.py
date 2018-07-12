import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import warnings


def simulate_gbm(S0, vols, corr, T=1, mu=0, n_steps=2, n_sim=100, show_plt=False, debug_calcs=False):
    """
    Runs a geometric brownian motion simulation
    :param S0: initial value of variables
    :param vols: volatilities
    :param corr: correlation matrix
    :param T: total time of simulation
    :param mu: expected return (0 for futures)
    :param n_steps: number of steps, including initial value
    :param n_sim: number of simulations
    :param show_plt: true to show a plot of the simulations
    :param debug_calcs: only for debugging and testing: makes all random variables equal to 1
    :return: a matrix of shape n_sim, n_steps, n_vars
    """
    if debug_calcs:
        warnings.warn("Debug calcs is true. Using fake random numbers equal to 1")
    sig = np.array(vols)
    if n_steps < 2:
        raise Exception("Minimum step is 2")
    dt = T / n_steps
    n_vars = len(S0)

    sim = np.zeros((n_sim, n_steps, n_vars))

    t = np.broadcast_to(np.matlib.repmat(
        np.linspace(0, T, n_steps), n_sim, 1
    ).reshape((n_sim, n_steps, 1)),
                        sim.shape)  # t is n_sim x n_steps
    W = np.random.standard_normal(size=(n_steps * n_sim, n_vars))
    if debug_calcs:
        W = np.ones(W.shape)
    # make them correlated
    R = np.linalg.cholesky(corr)
    eps = R @ W.T  # W is n_var x (n_sim*n_steps)
    eps = np.swapaxes(eps.reshape((n_vars, n_steps, n_sim)), 0, 2)  # eps is n_sim, n_steps, n_vars
    eps[(int(n_sim / 2)):, :, :] = -eps[:int(n_sim / 2), :, :]
    W = np.cumsum(eps, axis=1) * np.sqrt(dt)  ### standard brownian motion ###
    W[:, 0, :] = 0  # To make it coherent with dt that starts in 0
    X = (mu - 0.5 * sig ** 2) * t + sig * W
    # Wikipedia: step=np.exp((mu-sigma**2/2)*dt)*np.exp(sigma*np.random.normal(0,np.sqrt(dt),(1,n)))

    S = S0 * np.exp(X)  ### geometric brownian motion ###
    if show_plt:
        for i in range(S.shape[-1]):
            plt.plot(t[0, :, 0], np.swapaxes(S[:, :, i], 0, 1))
            # plot mean on top
            plt.plot(t[0, :, 0], S[:, :, i].mean(axis=0), linewidth=4, color='k')

        plt.show()
    return S


def calculate_margin(S, freeze_dim=-1):
    """
    simple margin calculator margin = power - fuel * production at delivery
    :param S: simulation matrix of 3 variables. First is power, second fuel third production
    :param freeze_dim: a list or tupple. If an index of a variable is included in the list
    the initial value
    :return:
    """

    if isinstance(freeze_dim, int):
        freeze_dim_iter = (freeze_dim, )
    else:
        freeze_dim_iter = freeze_dim

    vars = [S[:, -1 if i_var not in freeze_dim_iter else 0, 0] for i_var in range(S.shape[2])]
    if len(vars) == 3:
        power, fuel, production = vars
        margin = (power - fuel) * production
    else:
        power, fuel = vars
        margin = (power - fuel) * 100
    return margin


def compute_margin_at_risk(S, freeze_dim=-1):
    margin = calculate_margin(S, freeze_dim)
    indexes = np.argsort(margin)
    p5_indexes = indexes[:1 + int(np.ceil(len(margin) * .05))]
    p5_margin = margin[p5_indexes[-1]]
    avg_margin = margin.mean(axis=0)

    p5_margin = numpy.percentile(margin, 5, axis=0, interpolation="nearest")

    plt.clf()
    plt.hist(margin)
    plt.show()
    MaR = avg_margin - p5_margin
    print(f"Expected margin: {avg_margin}, Margin at risk: {MaR}")
    return MaR


def check_statistical_properties(S, T):
    """
    Calculates volatilites and correlation (anualized) of the Sample S
    :param T: total time of simulation
    :param S: n_sim x n_step x n_vars
    :return: None
    """

    # compute returns
    logreturns = np.log(S[:, :-1, :] / S[:, 1:, :])
    anualized_vols = np.std(logreturns, axis=1).mean(axis=0) * np.sqrt(S.shape[1] / T)
    print(f"Anualized vols: {anualized_vols}")
    # correlation
    corr = []
    for i in range(S.shape[2]):
        for j in range(i + 1, S.shape[2]):
            corr_sim = 0
            for i_sim in range(S.shape[0]):
                corr_sim += np.corrcoef(logreturns[i_sim, :, i],
                                       logreturns[i_sim, :, j], rowvar=False)[1, 0]
            corr.append(corr_sim / S.shape[0])
    print(f"Correlations: {corr}")


def main():
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

    corr = np.array([[1, rho12, rho13],
                     [rho12, 1, rho23],
                     [rho13, rho23, 1]])

    T = 1
    S = simulate_gbm(S0, vols, corr, T=T, mu=0, n_steps=50, n_sim=1000, show_plt=True)
    #S = simulate_gbm(S0, vols, corr, T=T, mu=0, n_steps=50, n_sim=1000)

    # S0 = S0[:2]
    # vols = vols[:2]
    # corr = corr[:2, :2]
    #
    # print(f"Input vols: {vols}")
    # print(f"Input corr: {corr}")
    #
    # #S = simulate_gbm(S0, vols, corr, T=T, mu=0, n_steps=50, n_sim=10, debug_calcs=True)
    # S = simulate_gbm(S0, vols, corr, T=T, mu=0, n_steps=50, n_sim=10)

    check_statistical_properties(S, T)

    compute_margin_at_risk(S)
    factors = []
    for i in range(len(S0)):
        factors.append(compute_margin_at_risk(S, i))
    print(factors)


if __name__ == "__main__":
    main()
