import tensorflow as tf
import numpy as np

class BachelorThesisOptimizer(tf.Module): # CHECK THIS PLEASE
    def __init__(self, learning_rate=np.ones(59)*1e-3, shape=59, convergence_rate=1.0001):
        self.learning_rate = tf.constant(
            learning_rate, dtype=tf.float64)
        self.lr_multiplier = tf.ones(
            dtype=tf.float64, shape=tf.shape(self.learning_rate)) * 8
        self.sign_last_grad = tf.zeros(
            dtype=tf.float64, shape=tf.shape(self.learning_rate))
        self.how_many_times_was_the_multiplier_1 = tf.zeros(
            dtype=tf.float64, shape=tf.shape(self.learning_rate))
        self.convergence_rate = convergence_rate

    def apply_gradients(self, elem):
        elem = elem[0]
        grads, var = elem

        if tf.math.count_nonzero(grads) == 1:
            self.lr_multiplier = tf.where(tf.less(tf.sign(grads * self.sign_last_grad), 0),
                                          tf.ones_like(
                self.lr_multiplier) * 0.5,
                self.lr_multiplier)
        else:
            self.lr_multiplier = tf.where(tf.less(tf.sign(grads * self.sign_last_grad), 0),  # Did the sign change?
                                          tf.ones_like(
                                              self.lr_multiplier) * 0.5,  # If yes, then 0.5
                                          tf.where(tf.equal(self.lr_multiplier, 0.5),  # If no, then is the multiplyer 0.5?
                                                   tf.ones_like(
                                                       self.lr_multiplier),  # If yes, then set it to 1 to see if minimum moved
                                                   # If no, then minimum moved so increase stepsize again
                                                   tf.ones_like(self.lr_multiplier) *
                                                   (2 / self.convergence_rate ** self.how_many_times_was_the_multiplier_1)))

            self.how_many_times_was_the_multiplier_1 = tf.where(tf.equal(self.lr_multiplier, 1),
                                                                self.how_many_times_was_the_multiplier_1 + 1,
                                                                self.how_many_times_was_the_multiplier_1)

        # The variables that are not being learned have zero gradient
        self.learning_lr_multiplierrate_multiplier = tf.where(
            tf.equal(grads, 0), tf.zeros_like(self.lr_multiplier), self.lr_multiplier)

        self.learning_rate = self.learning_rate * self.lr_multiplier

        var.assign_sub(self.learning_rate * tf.sign(grads))

        self.sign_last_grad = tf.sign(grads)


# class Symetric_Convex_optimizer(tf.Module):
#     def __init__(self, initial_step_size=1e-3, shape=59, convergence_rate=1.0001):
#         self.initial_step_size = tf.constant(
#             initial_step_size, dtype=tf.float64, shape=(shape))
#         self.lr_multiplier = tf.ones(
#             dtype=tf.float64, shape=(shape)) * 8
#         self.sign_last_grad = tf.zeros(
#             dtype=tf.float64, shape=(shape))
#         self.how_many_times_was_the_multiplier_1 = tf.zeros(
#             dtype=tf.float64, shape=(shape))
#         self.convergence_rate = convergence_rate
#         self.last_loss_value = tf.constant(0., dtype=tf.float64)
#         self.last_param_value = tf.zeros(
#             dtype=tf.float64, shape=(shape))

#     def apply_gradients(self, grads, var, loss_value):

#         if (tf.equal(self.last_loss_value, 0.)):
#             self.last_param_value = tf.identity(var)
#             var.assign_sub(self.initial_step_size * tf.sign(grads))

#         else:

#         # self.sign_last_grad = tf.sign(grads)

#         if tf.math.count_nonzero(grads) == 1:
#             self.lr_multiplier = tf.where(tf.less(tf.sign(grads * self.sign_last_grad), 0),
#                                           tf.ones_like(
#                 self.lr_multiplier) * 0.5,
#                 self.lr_multiplier)
#         else:
#             self.lr_multiplier = tf.where(tf.less(tf.sign(grads * self.sign_last_grad), 0),  # Did the sign change?
#                                           tf.ones_like(
#                                               self.lr_multiplier) * 0.5,  # If yes, then 0.5
#                                           tf.where(tf.equal(self.lr_multiplier, 0.5),  # If no, then is the multiplyer 0.5?
#                                                    tf.ones_like(
#                                                        self.lr_multiplier),  # If yes, then set it to 1 to see if minimum moved
#                                                    # If no, then minimum moved so increase stepsize again
#                                                    tf.ones_like(self.lr_multiplier) *
#                                                    (2 / self.convergence_rate ** self.how_many_times_was_the_multiplier_1)))

#             self.how_many_times_was_the_multiplier_1 = tf.where(tf.equal(self.lr_multiplier, 1),
#                                                                 self.how_many_times_was_the_multiplier_1 + 1,
#                                                                 self.how_many_times_was_the_multiplier_1)

#         # The variables that are not being learned have zero gradient
#         self.learning_lr_multiplierrate_multiplier = tf.where(
#             tf.equal(grads, 0), tf.zeros_like(self.lr_multiplier), self.lr_multiplier)

#         self.initial_step_size = self.initial_step_size * self.lr_multiplier

#         var.assign_sub(self.initial_step_size * tf.sign(grads))

#         self.sign_last_grad = tf.sign(grads)


class BachelorThesisOptimizer_with_schedule(tf.Module):
    def __init__(self, learning_rate=1e-3, shape=59, convergence_rate=1.0001, amount_threshold=1e-6):
        self.initial_lr = tf.constant(learning_rate, dtype=tf.float64)
        self.learning_rate = tf.constant(
            learning_rate / 8 * tf.ones(shape=(shape,), dtype=tf.float64), dtype=tf.float64)
        self.lr_multiplier = tf.constant(
            tf.ones(shape=(shape,), dtype=tf.float64) * 8, dtype=tf.float64)
        self.sign_last_grad = tf.constant(
            tf.zeros(shape=(shape,), dtype=tf.float64), dtype=tf.float64)
        self.how_many_times_was_the_multiplier_1 = tf.constant(
            tf.zeros(shape=(shape,), dtype=tf.float64), dtype=tf.float64)
        self.convergence_rate = convergence_rate

        # New variables for tracking changes
        self.iteration_counter = tf.constant(0, dtype=tf.int32)
        self.var_10_steps_ago = tf.constant(
            tf.zeros(shape=(shape,), dtype=tf.float64), dtype=tf.float64)
        self.amount_threshold = tf.constant(amount_threshold, dtype=tf.float64)

    def apply_gradients(self, elem):
        elem = elem[0]
        grads, var = elem

        if tf.math.count_nonzero(grads) == 1:
            self.lr_multiplier.assign(tf.where(
                tf.less(tf.sign(grads * self.sign_last_grad), 0),
                tf.ones_like(self.lr_multiplier) * 0.5,
                self.lr_multiplier))
        else:
            self.lr_multiplier = tf.where(tf.less(tf.sign(grads * self.sign_last_grad), 0),  # Did the sign change?
                                          tf.ones_like(
                                              self.lr_multiplier) * 0.5,  # If yes, then 0.5
                                          tf.where(tf.equal(self.lr_multiplier, 0.5),  # If no, then is the multiplyer 0.5?
                                                   tf.ones_like(
                                                       self.lr_multiplier),  # If yes, then set it to 1 to see if minimum moved
                                                   # If no, then minimum moved so increase stepsize again
                                                   tf.ones_like(self.lr_multiplier) *
                                                   (2 / self.convergence_rate ** self.how_many_times_was_the_multiplier_1)))

            self.how_many_times_was_the_multiplier_1 = tf.where(tf.equal(self.lr_multiplier, 1),
                                                                self.how_many_times_was_the_multiplier_1 + 1,
                                                                self.how_many_times_was_the_multiplier_1)

        # Update learning rate multiplier for non-zero gradients
        self.learning_lr_multiplierrate_multiplier = tf.where(
            tf.equal(grads, 0), tf.zeros_like(self.lr_multiplier), self.lr_multiplier)

        # Update learning rate
        self.learning_rate = self.learning_rate * self.lr_multiplier

        # Update variable
        var.assign_sub(self.learning_rate * tf.sign(grads))

        # Update last gradient sign
        self.sign_last_grad = tf.sign(grads)

        # Increment iteration counter
        self.iteration_counter = self.iteration_counter + 1

        # Check if we have reached 10 iterations
        if self.iteration_counter >= 10:
            # Compute the change
            delta_var = tf.abs(var - self.var_10_steps_ago)
            max_delta = tf.reduce_max(delta_var)

            # Check if change is less than amount_threshold
            if max_delta < self.amount_threshold:
                tf.print("Resetting optimizer")
                # Reset lr_multiplier to ones
                self.lr_multiplier = tf.ones_like(self.lr_multiplier)

                self.initial_lr = self.initial_lr * 0.1
                self.learning_rate = tf.constant(
                    self.initial_lr / 8 * tf.ones_like(self.learning_rate))

                self.sign_last_grad = tf.zeros_like(self.sign_last_grad)

                self.how_many_times_was_the_multiplier_1 = tf.zeros_like(
                    self.how_many_times_was_the_multiplier_1)

                # Adjust amount_threshold
                self.amount_threshold = self.amount_threshold * 0.1

            # Update var_10_steps_ago
            self.var_10_steps_ago = tf.identity(var)

            # Reset iteration counter
            self.iteration_counter = 0


class BachelorThesisOptimizer_with_schedule_and_noise(tf.Module):
    def __init__(self, learning_rate=1e-3, shape=59, convergence_rate=1.01, amount_threshold=1e-6):
        self.initial_lr = tf.constant(learning_rate, dtype=tf.float64)
        self.learning_rate = tf.constant(
            learning_rate / 8 * tf.ones(shape=(shape,), dtype=tf.float64), dtype=tf.float64)
        self.lr_multiplier = tf.constant(
            tf.ones(shape=(shape,), dtype=tf.float64) * 8, dtype=tf.float64)
        self.sign_last_grad = tf.constant(
            tf.zeros(shape=(shape,), dtype=tf.float64), dtype=tf.float64)
        self.how_many_times_was_the_multiplier_1 = tf.constant(
            tf.zeros(shape=(shape,), dtype=tf.float64), dtype=tf.float64)
        self.convergence_rate = convergence_rate

        # Variables for tracking changes
        self.amount_threshold = tf.constant(amount_threshold, dtype=tf.float64)

        # Buffer to store variable values over the last 10 iterations
        self.var_buffer = []

        # Additional variables for the new behavior
        self.threshold_reached_times = tf.constant(0, dtype=tf.int32)
        self.stored_var = tf.constant(
            tf.zeros(shape=(shape,), dtype=tf.float64), dtype=tf.float64)

    def reset_optimizer(self):
        tf.print("Resetting optimizer")
        self.lr_multiplier = tf.ones_like(self.lr_multiplier)

        self.learning_rate = tf.constant(
            self.initial_lr / 8 * tf.ones_like(self.learning_rate))

        self.sign_last_grad = tf.zeros_like(self.sign_last_grad)

        self.how_many_times_was_the_multiplier_1 = tf.zeros_like(
            self.how_many_times_was_the_multiplier_1)

    def apply_gradients(self, elem):
        elem = elem[0]
        grads, var = elem

        if tf.math.count_nonzero(grads) == 1:
            self.lr_multiplier.assign(tf.where(
                tf.less(tf.sign(grads * self.sign_last_grad), 0),
                tf.ones_like(self.lr_multiplier) * 0.5,
                self.lr_multiplier))
        else:
            self.lr_multiplier = tf.where(tf.less(tf.sign(grads * self.sign_last_grad), 0),  # Did the sign change?
                                          tf.ones_like(
                                              self.lr_multiplier) * 0.5,  # If yes, then 0.5
                                          tf.where(tf.equal(self.lr_multiplier, 0.5),  # If no, then is the multiplyer 0.5?
                                                   tf.ones_like(
                                                       self.lr_multiplier),  # If yes, then set it to 1 to see if minimum moved
                                                   # If no, then minimum moved so increase stepsize again
                                                   tf.ones_like(self.lr_multiplier) *
                                                   (2 / self.convergence_rate ** self.how_many_times_was_the_multiplier_1)))

            self.how_many_times_was_the_multiplier_1 = tf.where(tf.equal(self.lr_multiplier, 1),
                                                                self.how_many_times_was_the_multiplier_1 + 1,
                                                                self.how_many_times_was_the_multiplier_1)

        self.learning_lr_multiplierrate_multiplier = tf.where(
            tf.equal(grads, 0), tf.zeros_like(self.lr_multiplier), self.lr_multiplier)

        self.learning_rate = self.learning_rate * self.lr_multiplier

        var.assign_sub(self.learning_rate * tf.sign(grads))

        self.sign_last_grad = tf.sign(grads)

        self.var_buffer.append(tf.identity(var))

        if len(self.var_buffer) > 10:

            var_10_steps_ago = self.var_buffer.pop(0)

            difference_between_current_var_and_var_10_steps_ago = tf.reduce_max(
                tf.abs(var - var_10_steps_ago))

            # Check if change is less than amount_threshold
            if difference_between_current_var_and_var_10_steps_ago < self.amount_threshold:
                if self.threshold_reached_times == 0:
                    # First time threshold is reached
                    tf.print("Threshold reached, adding noise to variable")
                    # Store the current variable
                    self.stored_var = tf.identity(var)
                    # Generate noise
                    noise = tf.random.normal(
                        shape=var.shape, mean=0.0, stddev=self.initial_lr, dtype=tf.float64)
                    noise = tf.cast(noise, var.dtype)

                    var.assign_add(noise)

                    self.threshold_reached_times = tf.constant(
                        1, dtype=tf.int32)

                    self.reset_optimizer()

                elif self.threshold_reached_times == 1:
                    tf.print(
                        "Threshold reached a second time. Taking the mean")

                    self.reset_optimizer()

                    var.assign((var + self.stored_var) / 2)
                    self.threshold_reached_times = tf.constant(
                        2, dtype=tf.int32)

                else:
                    tf.print(
                        "Threshold reached a third time. proceeding optimizer")

                    self.initial_lr = self.initial_lr * 0.1
                    self.amount_threshold = self.amount_threshold * 0.1
                    self.reset_optimizer()

                    self.threshold_reached_times = tf.constant(
                        0, dtype=tf.int32)


class BachelorThesisOptimizerWithRelu(tf.Module):
    def __init__(self, learning_rate=1e-3, shape=59, convergence_rate=1.01):
        self.learning_rate = tf.constant(
            learning_rate, dtype=tf.float64, shape=(shape))
        self.lr_multiplier = tf.ones(
            dtype=tf.float64, shape=(shape))
        self.sign_last_grad = tf.zeros(
            dtype=tf.float64, shape=(shape))
        self.how_many_times_was_the_multiplier_1 = tf.zeros(
            dtype=tf.float64, shape=(shape))
        self.convergence_rate = convergence_rate

    def apply_gradients(self, grads, var):
        if tf.math.count_nonzero(grads) == 1:
            self.lr_multiplier = tf.where(tf.less(tf.sign(grads * self.sign_last_grad), 0),
                                          tf.ones_like(
                self.lr_multiplier) * 0.5,
                self.lr_multiplier)
        else:
            self.lr_multiplier = tf.where(tf.less(tf.sign(grads * self.sign_last_grad), 0),
                                          tf.ones_like(
                self.lr_multiplier) * 0.5,
                tf.where(tf.equal(self.lr_multiplier, 0.5),
                         tf.ones_like(
                    self.lr_multiplier), tf.ones_like(self.lr_multiplier) * (2 / self.convergence_rate ** self.how_many_times_was_the_multiplier_1)))

            self.how_many_times_was_the_multiplier_1 = tf.where(tf.equal(self.lr_multiplier, 1),
                                                                self.how_many_times_was_the_multiplier_1 + 1,
                                                                self.how_many_times_was_the_multiplier_1)

        # The variables that are not being learned have zero gradient
        self.learning_lr_multiplierrate_multiplier = tf.where(
            tf.equal(grads, 0), tf.zeros_like(self.lr_multiplier), self.lr_multiplier)

        self.learning_rate = self.learning_rate * self.lr_multiplier

        var.assign(tf.maximum(var - self.learning_rate * tf.sign(grads), 0.0))

        self.sign_last_grad = tf.sign(grads)
