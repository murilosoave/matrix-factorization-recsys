import numpy as np
import pytest

from typing import Tuple

from src.matrix_factorization import MatrixFactorization


class TestMatrixFactorization:
    @pytest.fixture
    def matrix_factorization(self):
        """Create an instance of MatrixFactorization for testing."""
        return MatrixFactorization()

    def test_fit(self, matrix_factorization):
        """Test that the fit method returns the expected output."""
        data = np.array([[4, 0, 0, 0, 0],
                        [0, 5, 0, 0, 0],
                        [0, 0, 3, 0, 0],
                        [0, 0, 0, 2, 0],
                        [0, 0, 0, 0, 5]])
        users_latent_features, items_latent_features = matrix_factorization.fit(data)

        assert users_latent_features.shape == (5, 5)
        assert items_latent_features.shape == (5, 5)
        assert isinstance(users_latent_features, np.ndarray)
        assert isinstance(items_latent_features, np.ndarray)

    def test_update_weights(self, matrix_factorization):
        """Test that the _update_weights method returns the expected output."""
        data = np.array([[4, 0, 0, 0, 0],
                        [0, 5, 0, 0, 0],
                        [0, 0, 3, 0, 0],
                        [0, 0, 0, 2, 0],
                        [0, 0, 0, 0, 5]])
        users_latent_features = np.random.rand(5, 5)
        items_latent_features = np.random.rand(5, 5)

        updated_users_latent_features, updated_items_latent_features = matrix_factorization._update_weights(data, users_latent_features, items_latent_features)

        assert updated_users_latent_features.shape == (5, 5)
        assert updated_items_latent_features.shape == (5, 5)
        assert isinstance(updated_users_latent_features, np.ndarray)
        assert isinstance(updated_items_latent_features, np.ndarray)

    def test_compute_gradient(self, matrix_factorization):
        """Test that the _compute_gradient method returns the expected output."""
        error = 2
        users_latent_features = np.random.rand(5, 5)
        items_latent_features = np.random.rand(5, 5)
        user_idx = 2
        item_idx = 3
        latent_idx = 4

        updated_users_latent_features, updated_items_latent_features = matrix_factorization._compute_gradient(error, users_latent_features, items_latent_features, user_idx, item_idx, latent_idx)

        assert updated_users_latent_features.shape == (5, 5)
        assert updated_items_latent_features.shape == (5, 5)
        assert isinstance(updated_users_latent_features, np.ndarray)
        assert isinstance(updated_items_latent_features, np.ndarray)