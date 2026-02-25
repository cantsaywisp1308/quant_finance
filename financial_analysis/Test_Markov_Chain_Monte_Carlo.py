import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import scipy.stats as stats

plt.style.use("ggplot")


def create_mcmc_model(alpha, beta, n, z, iterations):
    # Use PyMC to construct a model context
    with pm.Model() as basic_model:
        # Define our prior belief about the fairness
        # of the coin using a Beta distribution
        theta = pm.Beta("theta", alpha=alpha, beta=beta)

        # Define the Bernoulli likelihood function
        y = pm.Binomial("y", n=n, p=theta, observed=z)

        # Carry out the MCMC analysis using the Metropolis algorithm
        # Use Maximum A Posteriori (MAP) optimisation as initial value for MCMC
        start = pm.find_MAP()

        # Use the Metropolis algorithm (as opposed to NUTS or HMC, etc.)
        step = pm.Metropolis()

        # Calculate the trace
        trace = pm.sample(
            draws=iterations,
            step=step,
            init=start,
            chains=1,
            random_seed=1,
            progressbar=True
        )
    return trace


def plot_mcmc_comparison(trace, bins, alpha, beta, alpha_post, beta_post):
    # Plot the posterior histogram from MCMC analysis
    plt.hist(
        trace['posterior']['theta'][0].to_numpy(), bins,
        histtype="step", density=True,
        label="Posterior (MCMC)", color="red"
    )

    # Plot the analytic prior and posterior beta distributions
    x = np.linspace(0, 1, 100)
    plt.plot(
        x, stats.beta.pdf(x, alpha, beta),
        "--", label="Prior", color="blue"
    )
    plt.plot(
        x, stats.beta.pdf(x, alpha_post, beta_post),
        label='Posterior (Analytic)', color="green"
    )

    # Update the graph labels
    plt.legend(title="Parameters", loc="best")
    plt.xlabel("$\\theta$, Fairness")
    plt.ylabel("Density")
    plt.show()

    # Show the trace plot
    pm.traceplot(trace)
    plt.show()


if __name__ == '__main__':

    # Parameter values for prior and analytic posterior
    n = 50
    z = 10
    alpha = 12
    beta = 12
    alpha_post = 22
    beta_post = 52

    # How many iterations of the Metropolis
    # algorithm to carry out for MCMC
    iterations = 100000

    # Number of Bins for Histogram
    bins=50

    mcmc_model = create_mcmc_model(alpha, beta, n, z, iterations)
    plot_mcmc_comparison(mcmc_model, bins, alpha, beta, alpha_post, beta_post)