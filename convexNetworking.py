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
        input_dim = input_shape[-1]

        # Weights for normal vectors (num_hyperplanes, input_dim)
        # Each row is the normal vector for one hyperplane
        w_init = tf.random_normal_initializer()
        self.normal_vectors = tf.Variable(
            initial_value=w_init(shape=(self.num_hyperplanes, input_dim), dtype="float32"),
            trainable=True,
            name="normal_vectors"
        )

        # Biases for each hyperplane (num_hyperplanes, 1)
        b_init = tf.zeros_initializer()
        self.biases = tf.Variable(
            initial_value=b_init(shape=(self.num_hyperplanes, 1), dtype="float32"),
            trainable=True,
            name="biases"
        )

        # Optional: Add non-trainable boundary hyperplanes if needed (e.g., for a bounding box)
        # This would require defining their fixed normal vectors and biases here
        # For simplicity, this example only uses trainable hyperplanes.

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
        # inputs shape: (batch_size, input_dim) - assuming 2D input for simplicity in dot product
        # normal_vectors shape: (num_hyperplanes, input_dim)
        # biases shape: (num_hyperplanes, 1)

        # Expand inputs to (batch_size, 1, input_dim) for broadcasting with normal_vectors
        expanded_inputs = tf.expand_dims(inputs, axis=-2) # Shape (batch_size, 1, input_dim)

        # Calculate dot product for each hyperplane and each input point
        # (batch_size, 1, input_dim) * (1, num_hyperplanes, input_dim) -> element-wise multiplication
        # Sum over the last dimension (input_dim) to get dot products
        # Result shape: (batch_size, num_hyperplanes)
        dot_products = tf.reduce_sum(expanded_inputs * tf.expand_dims(self.normal_vectors, axis=0), axis=-1)

        # Add biases. Biases shape (num_hyperplanes, 1), need to broadcast to (batch_size, num_hyperplanes)
        # Expand biases to (1, num_hyperplanes, 1) and then squeeze the last dim after adding
        signed_distances = dot_products + tf.squeeze(tf.expand_dims(self.biases, axis=0), axis=-1) # Shape (batch_size, num_hyperplanes)

        # The point (x) is inside the half-space defined by w.x + b >= 0 if w is the outward normal.
        # If w is the inward normal, the condition is w.x + b <= 0.
        # Let's assume w are inward normals, so we want w.x + b <= 0 for all hyperplanes.
        # This is equivalent to -(w.x + b) >= 0 for all hyperplanes.
        # We can use a sigmoid or a step function on -(w.x + b).
        # A common approach for convex hulls is to take the minimum of the signed distances
        # (assuming inward normals and normalized distances). If the minimum is >= 0, the point is inside.
        # For a differentiable approach suitable for training, we can use a smooth approximation
        # like a sigmoid or a product of sigmoids.

        # Option 1: Product of Sigmoids (smooth approximation of AND)
        # We want sigmoid(-(w.x + b)) to be close to 1 for each hyperplane.
        # The product of these sigmoids will be close to 1 only if all are close to 1.
        # Use a scaling factor 's' for the sigmoid for steeper transitions.
        s = 50.0 # Scaling factor - higher means steeper
        half_space_scores = tf.sigmoid(-signed_distances * s) # Shape (batch_size, num_hyperplanes)

        # Multiply scores for each hyperplane for a given point
        # The result is close to 1 if the point is on the correct side of ALL hyperplanes
        hull_score = tf.reduce_prod(half_space_scores, axis=-1, keepdims=True) # Shape (batch_size, 1)

        # Option 2: Minimum of (normalized) signed distances (less common in standard NN, but conceptually clear)
        # Requires careful normalization of normal vectors and biases.

        # Let's stick with Option 1 for this example as it's more common in end-to-end training.

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

# --- Example Usage ---

# Create some dummy data (e.g., points inside and outside a cube in 3D)
# Input dim = 3
# A cube can be defined by 6 hyperplanes (x>=0, x<=1, y>=0, y<=1, z>=0, z<=1)
# Let's try to learn a hull that approximates a region.
# For simplicity, let's try a region around the origin in 3D.
# Points near (0,0,0) are class 1 (inside), others are class 0 (outside).

# Data generation (similar to your notebooks)
np.random.seed(42)
num_samples = 1000
input_dim = 3
# Points inside a small cube around origin
xs_inside = np.random.rand(num_samples // 2, input_dim) * 0.2 - 0.1 # Centered around 0, small range
ys_inside = np.ones((num_samples // 2, 1), dtype=np.float32)

# Points outside
xs_outside = np.random.rand(num_samples // 2, input_dim) * 2.0 - 1.0 # Larger range
# Ensure some outside points are not inside the small cube
outside_mask = np.any(np.abs(xs_outside) > 0.1, axis=1)
xs_outside = xs_outside[outside_mask]
ys_outside = np.zeros((xs_outside.shape[0], 1), dtype=np.float32)

# Combine data
xs = np.concatenate([xs_inside, xs_outside], axis=0)
ys = np.concatenate([ys_inside, ys_outside], axis=0)

# Shuffle data
indices = np.arange(xs.shape[0])
np.random.shuffle(indices)
xs = xs[indices]
ys = ys[indices]

# Define the model using the custom layer
# Let's try to learn 6 hyperplanes for a 3D space
num_hyperplanes_to_learn = 6

model = keras.Sequential([
    # Input layer - specify input shape
    keras.layers.InputLayer(input_shape=(input_dim,)),
    # Add the custom convex hull layer
    InConvexHull(num_hyperplanes=num_hyperplanes_to_learn),
    # Optional: Add a final Dense layer with sigmoid if you want output strictly between 0 and 1
    # keras.layers.Dense(1, activation='sigmoid') # The custom layer already outputs 0-1 score
])

# Compile the model
# Use BinaryCrossentropy as the loss function, as the output is a score/probability
# The custom layer's output is already between 0 and 1 due to the product of sigmoids.
model.compile(optimizer='Adam',
              loss='BinaryCrossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
# You'll likely need more epochs and potentially tune hyperparameters
# (learning rate, number of hyperplanes, sigmoid scaling 's')
# The data generation here is also very simple; real data will be more complex.
print("\nTraining the model...")
history = model.fit(xs, ys, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model (optional)
# loss, accuracy = model.evaluate(xs, ys, verbose=0)
# print(f"\nFinal Loss: {loss:.4f}, Final Accuracy: {accuracy:.4f}")

# You can inspect the learned hyperplane parameters
# learned_normal_vectors = model.layers[0].get_weights()[0]
# learned_biases = model.layers[0].get_weights()[1]
# print("\nLearned Normal Vectors:\n", learned_normal_vectors)
# print("\nLearned Biases:\n", learned_biases)

# To use this for gating, you would apply the trained model to new data points.
# model.predict(new_data_points) would give you a score for each point.
# You could then set a threshold on this score (e.g., score > 0.5) to classify points
# as inside or outside the learned hull.
