import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import numpy as np

# Define a custom Keras layer to represent the convex hull logic
class InConvexHull(tf.keras.layers.Layer):
    """
    A custom Keras layer that checks if input points are inside a convex hull
    defined by a set of learned hyperplanes.
    """
    def __init__(self, num_hyperplanes, **kwargs):
        """
        Initializes the layer.

        Args:
            num_hyperplanes: The number of hyperplanes defining the convex hull.
            **kwargs: Additional keyword arguments for the Layer base class.
        """
        super().__init__(**kwargs)
        self.num_hyperplanes = num_hyperplanes

    def build(self, input_shape):
        """
        Builds the layer, creating trainable weights for the hyperplanes.

        Args:
            input_shape: The shape of the input tensor (excluding batch size).
                         Expected shape is (..., input_dim).
        """
        if input_shape[-1] is None:
            raise ValueError("The last dimension of the inputs to `InConvexHull` "
                             "should be defined. Found None.")
        input_dim = input_shape[-1]

        # Weights for normal vectors (num_hyperplanes, input_dim)
        # Each row is the normal vector for one hyperplane (assuming inward normals)
        w_init = tf.random_normal_initializer()
        self.normal_vectors = tf.Variable(
            initial_value=w_init(shape=(self.num_hyperplanes, input_dim), dtype="float32"),
            trainable=True,
            name="normal_vectors"
        )

        # Biases for each hyperplane (num_hyperplanes,)
        b_init = tf.zeros_initializer()
        self.biases = tf.Variable(
            initial_value=b_init(shape=(self.num_hyperplanes,), dtype="float32"),
            trainable=True,
            name="biases"
        )

        super().build(input_shape)

    def call(self, inputs):
        """
        Computes the output of the layer.

        Args:
            inputs: The input tensor of points. Shape (batch_size, ..., input_dim).

        Returns:
            A tensor representing the "score" or "probability" of each point
            being inside the convex hull.
            Shape (batch_size, ..., 1).
        """
        # Ensure inputs are float32
        inputs = tf.cast(inputs, dtype=tf.float32)

        # Calculate signed distance to each hyperplane: w.x + b
        # inputs shape: (batch_size, input_dim) or (batch_size, ..., input_dim)
        # normal_vectors shape: (num_hyperplanes, input_dim)
        # biases shape: (num_hyperplanes,)

        # Reshape inputs to (batch_size, ..., 1, input_dim) for broadcasting with normal_vectors
        # This handles both (batch_size, input_dim) and higher dimensional inputs
        expanded_inputs = tf.expand_dims(inputs, axis=-2) # Shape (batch_size, ..., 1, input_dim)

        # Expand normal_vectors to (1, ..., num_hyperplanes, input_dim) for broadcasting
        expanded_normal_vectors = tf.expand_dims(self.normal_vectors, axis=0)
        while expanded_normal_vectors.shape.rank < expanded_inputs.shape.rank:
            expanded_normal_vectors = tf.expand_dims(expanded_normal_vectors, axis=0)
        # expanded_normal_vectors shape: (1, ..., num_hyperplanes, input_dim)

        # Calculate dot product for each hyperplane and each input point
        # (batch_size, ..., 1, input_dim) * (1, ..., num_hyperplanes, input_dim) -> element-wise multiplication
        # Sum over the last dimension (input_dim) to get dot products
        # Result shape: (batch_size, ..., num_hyperplanes)
        dot_products = tf.reduce_sum(expanded_inputs * expanded_normal_vectors, axis=-1)

        # Add biases. Biases shape (num_hyperplanes,), need to broadcast
        # Expand biases to (1, ..., num_hyperplanes) for broadcasting
        expanded_biases = tf.expand_dims(self.biases, axis=0)
        while expanded_biases.shape.rank < dot_products.shape.rank:
             expanded_biases = tf.expand_dims(expanded_biases, axis=0)
        # expanded_biases shape: (1, ..., num_hyperplanes)

        # Signed distances assuming inward normals: w.x + b. We want w.x + b <= 0, or -(w.x + b) >= 0
        signed_distances = dot_products + expanded_biases # Shape (batch_size, ..., num_hyperplanes)

        # Use a sigmoid on -(w.x + b). A high score (near 1) indicates the point is on the correct side of the hyperplane.
        # Use a scaling factor 's' for the sigmoid for steeper transitions.
        s = 50.0 # Scaling factor - higher means steeper
        half_space_scores = tf.sigmoid(-signed_distances * s) # Shape (batch_size, ..., num_hyperplanes)

        # Multiply scores for each hyperplane for a given point
        # The result is close to 1 only if the point is on the correct side of ALL hyperplanes (i.e., inside the hull)
        hull_score = tf.reduce_prod(half_space_scores, axis=-1, keepdims=True) # Shape (batch_size, ..., 1)

        return hull_score # Output is a score between 0 and 1

    def get_config(self):
        """
        Returns the config of the layer. Required for saving/loading models.
        """
        config = super().get_config()
        config.update({
            "num_hyperplanes": self.num_hyperplanes,
        })
        return config

# Example Usage (will be used in test.ipynb)
# Data generation (similar to your notebooks)
# np.random.seed(42)
# num_samples = 1000
# input_dim = 3
# # Points inside a small cube around origin
# xs_inside = np.random.rand(num_samples // 2, input_dim) * 0.2 - 0.1 # Centered around 0, small range
# ys_inside = np.ones((num_samples // 2, 1), dtype=np.float32)

# # Points outside
# xs_outside = np.random.rand(num_samples // 2, input_dim) * 2.0 - 1.0 # Larger range
# # Ensure some outside points are not inside the small cube
# outside_mask = np.any(np.abs(xs_outside) > 0.1, axis=1)
# xs_outside = xs_outside[outside_mask]
# ys_outside = np.zeros((xs_outside.shape[0], 1), dtype=np.float32)

# # Combine data
# xs = np.concatenate([xs_inside, xs_outside], axis=0)
# ys = np.concatenate([ys_inside, ys_outside], axis=0)

# # Shuffle data
# indices = np.arange(xs.shape[0])
# np.random.shuffle(indices)
# xs = xs[indices]
# ys = ys[indices]

# # Define the model using the custom layer
# # Let's try to learn 6 hyperplanes for a 3D space
# num_hyperplanes_to_learn = 6

# model = keras.Sequential([
#     # Input layer - specify input shape
#     keras.layers.InputLayer(input_shape=(input_dim,)),
#     # Add the custom convex hull layer
#     InConvexHull(num_hyperplanes=num_hyperplanes_to_learn),
# ])

# # Compile the model
# model.compile(optimizer='Adam',
#               loss='BinaryCrossentropy', # Use BinaryCrossentropy for score output
#               metrics=['accuracy'])

# # Print model summary
# model.summary()

# # Train the model
# print("\nTraining the model...")
# history = model.fit(xs, ys, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# # Evaluate the model (optional)
# # loss, accuracy = model.evaluate(xs, ys, verbose=0)
# # print(f"\nFinal Loss: {loss:.4f}, Final Accuracy: {accuracy:.4f}")

# # To use this for gating, you would apply the trained model to new data points.
# # model.predict(new_data_points) would give you a score for each point.
# # You could then set a threshold on this score (e.g., score > 0.5) to classify points
# # as inside or outside the learned hull.