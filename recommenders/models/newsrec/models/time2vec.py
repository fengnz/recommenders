"""module for implementing time to vector encoding"""

import tensorflow as tf
import tensorflow.keras.layers as KL


class Time2Vec(KL.Layer):
    """time2vector encoding layer"""

    def __init__(self, kernel: int = 64, activation: str = "sin") -> None:
        """
        Args:
            kernel (int, optional): length of time vector representation. Defaults to 64
            activation (str, optional): periodic activation for time encoding. Defaults to "sin".
        Raises:
            NotImplementedError: Non-supported activations
        """

        # periodic components
        if activation in ["sin", "cos"]:
            activation = {"sin": tf.math.sin, "cos": tf.math.cos}[activation]
        else:
            raise NotImplementedError(
                f"'{activation}' is an unsupported periodic activation."
            )

        super().__init__(trainable=True, name="Time2VecLayer_" + activation.__name__)

        self.k = kernel - 1
        self.p_activation = activation

    def build(self, input_shape: tuple) -> None:
        """method for building and initializing the weights for the tensor operations
        Args:
            input_shape (tuple): shape of the incoming tensor
        """
        # Linear component
        self.w_b = self.add_weight(
            shape=(1, input_shape[1], 1), initializer="uniform", trainable=True
        )

        self.b_b = self.add_weight(
            shape=(1, input_shape[1], 1), initializer="uniform", trainable=True
        )

        # Periodic components
        self.freq = self.add_weight(
            shape=(1, input_shape[1], self.k), initializer="uniform", trainable=True
        )

        self.phase = self.add_weight(
            shape=(1, input_shape[1], self.k), initializer="uniform", trainable=True
        )

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:

        """method to perform the layer operation
        Args:
            inputs (tf.Tensor): shape = (batch_size, feature_size)
        Returns:
            tf.Tensor: shape = (batch_size, feature_size, length of time vector representation)
        """

        inputs = tf.expand_dims(inputs, axis=-1)

        # Linear components
        lin = (
            # Multiply each time dimension with the corresponding linear time component
                tf.multiply(inputs, self.w_b)
                # Bias component for each time dimension
                + self.b_b
        )

        # Periodic components
        # Multiply each time dimension (M, D, H, mins, etc.) with the corresponding frequency vector
        per = tf.multiply(tf.tile(inputs, multiples=[1, 1, self.k]), self.freq)
        # Phase vector for each time dimension
        per = self.p_activation(per + self.phase)
        return tf.concat([lin, per], -1)

    def compute_output_shape(self, input_shape: tuple) -> tuple:
        """computes the shape of output tensor
        Args:
            input_shape (tuple): shape of incoming tensor
        Returns:
            tuple: shape of outgoing tensor
        """
        return (input_shape[0], input_shape[1], self.k + 1)


if __name__ == "__main__":
    # 32 samples with time represented as %%M%%D%%H%%S
    test_vector = tf.random.uniform(shape=(32, 4), dtype=tf.float32)
    xti = Time2Vec(16, "sin")(test_vector)
    # Shape = (32, 4, 16)! 16 dimensional representation for each t-dimension
    print(xti.shape)