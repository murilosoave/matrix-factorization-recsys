import numpy as np


class MatrixFactorization:
    def __init__(self,
                 n_latent_features=5,
                 n_iterations=5000,
                 learning_rate=0.001,
                 l2=0.01):
        """Initialize MatrixFactorization parameters."""
        self.n_latent_features = n_latent_features
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.l2 = l2
    
    def fit(self, data):
        "Fit a representation of users and items."
        users_count = data.shape[0]
        items_count = data.shape[1]
        
        users_latent_features = np.random.rand(users_count,
                                               self.n_latent_features)
        items_latent_features = np.random.rand(items_count,
                                               self.n_latent_features)
        
        for epoch in range(self.n_iterations):
            users_latent_features, items_latent_features = (
                self._update_weights(data, users_latent_features, items_latent_features)
            )

            # compute the loss.
            E = (data - users_latent_features.dot(items_latent_features.T))**2
            obj = (
                E[data.nonzero()].sum() +
                self.learning_rate *
                ((users_latent_features**2).sum() + (items_latent_features**2).sum())
            )
            if obj < 0.001:
                break
        
        print(obj)

        return users_latent_features, items_latent_features

    def _update_weights(self, data, users_latent_features, items_latent_features):
        for user_idx, item_idx in zip(*data.nonzero()):
            error = (
                data[user_idx, item_idx] -
                users_latent_features[user_idx,:].dot(items_latent_features[item_idx,:])
            )

            for latent_idx in range(self.n_latent_features):
                users_latent_features, items_latent_features = self._compute_gradient(
                  error,
                  users_latent_features,
                  items_latent_features,
                  user_idx,
                  item_idx,
                  latent_idx
                )


        return users_latent_features, items_latent_features
    
    def _compute_gradient(self,
                          error,
                          users_latent_features,
                          items_latent_features,
                          user_idx,
                          item_idx,
                          latent_idx):
        users_latent_features[user_idx][latent_idx] += (
            self.learning_rate *
            (2 * error * items_latent_features[item_idx][latent_idx] -
             self.l2/2 * users_latent_features[user_idx][latent_idx])
        )
        items_latent_features[item_idx][latent_idx] += (
            self.learning_rate *
            (2 * error * users_latent_features[user_idx][latent_idx] -
             self.l2/2 * items_latent_features[item_idx][latent_idx])
        )
        
        return users_latent_features, items_latent_features
        