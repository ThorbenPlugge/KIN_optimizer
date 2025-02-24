from typing import List
from dataclasses import dataclass
import queue
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('TkAgg')  # Or 'TkAgg'


def plot_thread(plot_queue, stop_event, plot_in_2D, zoombox, unknown_dimension, available_info_of_bodies):

    n = len(available_info_of_bodies)

    dimension = 2 if plot_in_2D else 3
    if unknown_dimension < 3:
        plot_in_2D = True

    # Create the plot axes
    if plot_in_2D:
        if unknown_dimension == 3:
            unknown_dimension = 2
        fig, ax = create_2D_plot()
        if zoombox == 'trappist':
            ax.set_xlim(-0.06, 0.06)
            ax.set_ylim(-0.06, 0.06)
        if zoombox == 'small':
            ax.set_xlim(-2500, 2500)
            ax.set_ylim(-2000, 2000)
        elif zoombox == 'smaller':
            ax.set_xlim(-500, 500)
            ax.set_ylim(-500, 500)
        elif zoombox == 'smallest':
            ax.set_xlim(-200, 200)
            ax.set_ylim(-200, 200)
    else:
        fig, ax = create_3D_plot()
        if zoombox == 'trappist':
            ax.set_xlim(-0.06, 0.06)
            ax.set_ylim(-0.06, 0.06)
            ax.set_zlim(-0.06, 0.06)
        if zoombox == 'small':
            ax.set_xlim(-2500, 2500)
            ax.set_ylim(-2000, 2000)
            ax.set_zlim(-2000, 2000)
        elif zoombox == 'smaller':
            ax.set_xlim(-500, 500)
            ax.set_ylim(-500, 500)
            ax.set_zlim(-500, 500)
        elif zoombox == 'smallest':
            ax.set_xlim(-200, 200)
            ax.set_ylim(-200, 200)
            ax.set_zlim(-200, 200)

    
    # Create line objects for each body
    lines = [ax.plot([], [], alpha=0.8, color='blue')[0] for _ in range(n)]
    if dimension == 3:
        lines = [ax.plot([], [], [], alpha=0.8, color='blue')[0] for _ in range(n)]

    # Draw static reference positions in red
    draw_reference_positions(ax, available_info_of_bodies,
                             dimension, unknown_dimension)

    # Update function for FuncAnimation
    def update(frame):
        try:
            data = plot_queue.get_nowait()  # Non-blocking get
            if data is None:  # Exit signal
                plt.close(fig)
                return
            t, j = data
            update_graph(ax, lines, n, t.numpy(), j,
                         dimension, unknown_dimension)
        except queue.Empty:
            pass  # No new data, keep the plot as is

        if dimension == 3:
            set_equal_aspect(ax)

    # Use FuncAnimation for smooth updates
    ani = FuncAnimation(fig, update, interval=50)  # Update every 50ms
    plt.show()


def draw_reference_positions(ax, available_info_of_bodies, dimension, unknown_dimension):
    """
    Draw static reference positions in red.

    Args:
        ax: Axes object for plotting.
        available_info_of_bodies: List of CelestialBody objects.
        dimension: 2 or 3, indicating the dimension of the plot.
        unknown_dimension: Dimension to exclude for 2D plots.
    """
    for body in available_info_of_bodies:
        for state in body.states:
            if state.time == -1:  # Skip guessed positions
                continue
            position = state.position
            if dimension == 2:
                temp = [i for i in range(3) if i != unknown_dimension]
                ax.scatter(position[temp[0]], position[temp[1]], color='red')
            else:
                ax.scatter(position[0], position[1], position[2], color='red')


def create_2D_plot():
    """Create and return a 2D plot figure and axes."""
    fig, ax = plt.subplots()
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Trajectories')
    ax.grid(True)
    ax.set_aspect('equal')
    return fig, ax


def create_3D_plot():
    """Create and return a 3D plot figure and axes."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('Trajectories')
    return fig, ax


def set_equal_aspect(ax):
    """
    Adjust 3D axes to have equal aspect ratio.

    Args:
        ax: Axes3D object to adjust.
    """
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    spans = limits[:, 1] - limits[:, 0]
    max_span = max(spans)
    centers = np.mean(limits, axis=1)
    new_limits = np.array([centers - max_span / 2, centers + max_span / 2]).T
    ax.set_xlim3d(new_limits[0])
    ax.set_ylim3d(new_limits[1])
    ax.set_zlim3d(new_limits[2])


def extract_positions(n, t):
    """
    Extract positions from the tensor t for each body over time.

    Args:
        n: Number of bodies.
        t: Tensor containing positions over time for each body.

    Returns:
        positions: A numpy array of shape (n, time_steps, 3).
    """
    positions = [[] for _ in range(n)]
    for state in t:
        for i in range(n):
            pos = state[i*3: i*3 + 3]
            positions[i].append(pos)
    return np.array(positions)


def update_graph(ax, lines, n, t, j, dimension=2, unknown_dimension=3):
    """
    Update the graph with new data without resetting the plot.

    Args:
        ax: The axes object to update.
        lines: List of line objects for each body.
        n: Number of bodies.
        t: Tensor containing positions over time for each body.
        j: Current training epoch (0-based).
        dimension: 2 or 3, indicating the dimension of the plot.
    """
    positions = extract_positions(n, t)

    temp = []
    for z in range(3):
        if z != unknown_dimension:
            temp.append(z)

    # Update trajectories
    for i, line in enumerate(lines):
        if dimension == 2:
            line.set_data(positions[i][:, temp[0]], positions[i][:, temp[1]])
        else:
            line.set_data_3d(positions[i][:, 0],
                             positions[i][:, 1], positions[i][:, 2])

    # Set the title with the current epoch
    ax.set_title(f'Training Epoch {j+1}')


def visualize_relevant_perturbations(r, v, tau, m, startingPosition, indices, num_points_considered_in_cost_function, num_steps_between_cost_func_points, perturbation_magnitude=0.1, steps=10):
    """
    Visualizes the effect of perturbing only the relevant elements (masses) in the input array on the function output.

    Parameters:
    - r: List of position vectors.
    - v: List of velocity vectors.
    - tau: Time step.
    - m: Masses array.
    - startingPosition: Dictionary of relevant celestial bodies and their starting values.
    - indices: List of indices in the mass array that correspond to the relevant celestial bodies.
    - num_points_considered_in_cost_function: Number of points to consider in cost function.
    - num_steps_between_cost_func_points: Steps between the points in the cost function.
    - perturbation_magnitude: Magnitude of perturbation.
    - steps: Number of perturbation steps to take.
    """
    # The relevant masses for perturbation are determined by the keys in the startingPosition dict
    relevant_bodies = list(startingPosition.keys())

    # Determine the layout for subplots (one plot per relevant body)
    num_elements = len(relevant_bodies)
    num_cols = int(np.ceil(np.sqrt(num_elements)))
    num_rows = int(np.ceil(num_elements / num_cols))

    # Create a figure with subplots for each relevant body
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 15))
    if num_elements == 1:
        axs = [axs]
    axs = np.array(axs).flatten()

    # For each relevant element, perturb and calculate function output
    for i, (body, index) in enumerate(zip(relevant_bodies, indices)):
        # Generate perturbations around the current element
        original_value = m[index]
        perturbations = np.linspace(
            original_value - perturbation_magnitude, original_value + perturbation_magnitude, steps)
        function_outputs = []

        for perturbed_value in perturbations:
            perturbed_array = m.copy()  # Create a perturbed version of the mass array
            # Apply perturbation to the relevant mass
            perturbed_array[index] = perturbed_value

            # Pass the perturbed_array to the cost function
            output = calculate_Loss(
                r, v, tau, perturbed_array, num_points_considered_in_cost_function, num_steps_between_cost_func_points)
            function_outputs.append(output)

        # Plot the perturbations vs function outputs for the current element
        axs[i].plot(perturbations, function_outputs, label=f'{body}')
        axs[i].set_xlabel(f'{body} Mass Perturbation')
        axs[i].set_ylabel('Function Output')
        axs[i].set_title(f'Perturbation of {body}')
        axs[i].grid(True)

    # Hide any extra subplots that aren't needed
    for j in range(num_elements, len(axs)):
        fig.delaxes(axs[j])

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


def calculate_Loss(r, v, tau, m, num_points_considered_in_cost_function, num_of_steps_in_do_step):
    """
    Determines the cost of mass by performing a number of steps and comparing
    the results to the provided positions and velocities (r, v).
    """
    r1 = copy.deepcopy(r[0])  # Start with initial position
    v1 = copy.deepcopy(v[0])  # Start with initial velocity
    total_cost = 0.0  # Initialize the total cost (squared difference)

    for i in range(1, num_points_considered_in_cost_function + 1):
        # Perform the steps for this point
        for _ in range(num_of_steps_in_do_step):
            r1, v1 = mc.do_step(tau, len(m), m, r1, v1)

        # Calculate the squared difference between r1, r[i] and v1, v[i]
        position_diff = np.sum((np.array(r1) - np.array(r[i])) ** 2)
        velocity_diff = np.sum((np.array(v1) - np.array(v[i])) ** 2)

        # Add the squared differences to the total cost
        total_cost += position_diff + velocity_diff

    return total_cost
