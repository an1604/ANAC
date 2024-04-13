class Learner:
    def __init__(self, prior):
        self.prior = prior
        self.likelihood = 0.
        self.posterior = 0.

    def learn(self, likelihood, sum_of_probs=1):
        self.likelihood = likelihood
        posterior = ((self.likelihood * self.prior) / sum_of_probs) * 10

        if posterior >= 1:
            self.posterior = posterior / 10
        else:
            self.posterior = posterior

        self.prior = self.posterior
        print(f"likelihood: {self.likelihood},"
              f"\nposterior: {self.posterior}")
