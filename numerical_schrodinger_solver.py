"""
Created by: Hamza Patwa

Python program that solves the Schroedinger equation using the Runge-Kutta 4th order method and the secant
root-searching method.

FLow of the program:
The function of the root search depends on phi left anf phi right.
phi left and phi right depend on the Runge-Kutta algorithm to create them.
The Runge-Kutta depends on the generalized velocity vector g().
g() depends on the second derivative of phi, which depends on the eigenenergy.

So, the root search depends on the eigenenergy.
The more accurate the eigenenergy, the closer secant gets to converging.
"""

import numpy as np
import math
import matplotlib.pyplot as plt

# Global constants
n_runge_kutta = 400  # Number of iterations for Runge-Kutta
n_secant = 10  # Number of iterations for secant
m = 1  # Every m indices, output of wavefunction is printed to the file
x1 = -10  # Initial x-value
x2 = 10.0  # Final x-value
h = (x2-x1) / n_runge_kutta  # Step size in x

# Wavefunctions.
# TODO: Pass the wavefunctions into all the functions instead of having them global
ul0 = np.empty(n_runge_kutta + 1)  # phi integrated from the left to the turning point
ul1 = np.empty(n_runge_kutta + 1)  # first derivative of phi integrated from the left to the turning point
ur0 = np.empty(n_runge_kutta + 1)  # phi integrated from the right to the turning point
ur1 = np.empty(n_runge_kutta + 1)  # first derivative of phi integrated from the right to the turning point
ul = np.empty(2)  # values of ul0[] and ul1[] at a particular point
ur = np.empty(2)  # values of ul0[] and ul1[] at a particular point
u = np.empty(n_runge_kutta + 1)  # The actual wavefunction phi, using ul for the left and ur for the right

countPrint = 0  # Just an extra counting variable used for printing stuff


# Calculates the quantity needed for secant to minimize
def f(e, V):
    # These are the global variables that will be edited
    global countPrint
    global ul0, ul1, ur0, ur1, ul, ur, u

    # Initial values
    ul0[0] = ul[0] = 0  # Left phi starts at 0
    ul1[0] = ul[1] = 0.01  # Left phi derivative starts at 0.01

    ur0[0] = ur[0] = 0  # Right phi starts at 0
    ur1[0] = ur[1] = -0.01  # Right phi derivative starts at -0.01

    # Calculates index where the left and right solutions will be matched together
    turnIndex = turning_index(e, V)

    # If the energy guess is, for example, a very small number, and the turning index gets set to 0, just
    # reset the index to 2, so that (turnIndex - 1) doesn't go out of bounds.
    if turnIndex == 0:
        turnIndex = 2

    # Printing stuff (only one time at the top)
    if countPrint == 0:
        print(f"index: {turnIndex}, x-val: {x1 + (turnIndex * h)}")
        print(f"Right solution goes from index 0 to {turnIndex+1}")
        print(f"Left solution goes from index {n_runge_kutta} to {turnIndex - 1}")
        countPrint = countPrint + 1

    # Runge Kutta for ul, from x1 -> turning point + 1
    for i in range(turnIndex + 2):
        x = x1 + (i * h)
        ul = rungeKutta(ul, x, e, h, V)
        ul0[i+1] = ul[0]
        ul1[i+1] = ul[1]

    # Runge Kutta for ur, from x2 -> turning point - 1
    for i in np.arange(n_runge_kutta, turnIndex - 2, -1):
        if i >= 0:
            x = x2 - ((n_runge_kutta - i) * h)
            ur = rungeKutta(ur, x, e, -h, V)
            ur0[i] = ur[0]
            ur1[i] = ur[1]

    # Rescale
    ul0 = ul0 * (ur0[turnIndex] / ul0[turnIndex])

    # Normalize
    u_squared = np.concatenate((ul0[:turnIndex]**2, ur0[turnIndex:]**2))
    u = np.concatenate((ul0[:turnIndex], ur0[turnIndex:]))  # Actual wavefunction with all the values
    norm_constant = math.sqrt(simpson(u_squared, h))
    u = u / norm_constant
    ul0 = ul0 / norm_constant
    ur0 = ur0 / norm_constant

    # Quantity that needs to be minimized, i.e., the subtraction of the derivative on the left and right sides
    return (ul0[turnIndex-1]-ul0[turnIndex+1]-ur0[turnIndex-1]+ur0[turnIndex+1]) / (2*h*ur0[turnIndex])


# Performs the root-searching algorithm on the function f and returns if there is a root, within a given tolerance
def secant(n, tolerance, x, dx, V):
    k = 0
    x1 = x + dx
    while math.fabs(dx) > tolerance and k < n:
        d = f(x1, V) - f(x, V)
        x2 = x1 - f(x1, V) * (x1 - x) / d
        x = x1
        x1 = x2
        dx = x1 - x
        k = k + 1

    if k == n:
        return None

    return x1


# Function used for numerical integration (integral of y with a step of h)
def simpson(y, h):
    n = np.size(y) - 1
    s0 = 0
    s1 = 0
    s2 = 0

    for i in np.arange(1, n, 2):
        s0 = s0 + y[i]
        s1 = s1 + y[i-1]
        s2 = s2 + y[i+1]

    s = (s1 + 4*s0 + s2) / 3

    if (n + 1) % 2 == 0:
        return h * (s + (5*y[n] + 8*y[n-1] - y[n-2]) / 12)
    return h*s


# Performs one Runge-Kutta step
# u is the 2-component numpy array that represents the wavefunction and its derivative at one particular x value
# x is the x-value
# e is the energy eigenvalue guess
# dx is the step size in x
# V is the potential function
def rungeKutta(u, x, e, dx, V):
    c1 = g(u, x, e, V)

    c2 = u + (dx * c1/2)
    c2 = g(c2, x + (dx/2), e, V)

    c3 = u + (dx * c2/2)
    c3 = g(c3, x + (dx/2), e, V)

    c4 = u + (dx * c3)
    c4 = g(c4, x + dx, e, V)

    return u + (dx * ((c1 + (2 * (c2 + c3)) + c4) / 6))


# Generalized velocity vector
# y is a 2-component np array that contains the two dynamical variables, the wavefunction and derivative in this case.
# x is the x-value.
# e is the energy eigenvalue guess.
def g(y, x, e, V):
    return np.array([y[1], (-2 * y[0]) * (e - V(x))])


# Potential function (Harmonic oscillator)
def V_harmonic(x):
    return 0.5 * math.pow(x, 2)


# Potential function (linear finite well)
def V_triangular_well(x):
    if x < 0 or x > 5:
        return 50
    return 10*x


# Calculates the index where the two solutions should be matched. This is the classical turning point, where V(x) = E
def turning_index(e, V):
    for i in range(n_runge_kutta):
        xr = x1 + (i * h)
        if xr > 0.0 and (e - V(xr)) <= 0.1:
            return i

    return 0


# Calculates the number of nodes in the global wavefunction variable u
def num_nodes():
    u_prev = u[1]
    node_count = 0
    for val in u[2:-1]:
        if val == 0.0:
            node_count = node_count + 1

        if val * u_prev < 0.0:
            node_count = node_count + 1

        u_prev = val

    return node_count


def main():
    tolerance = 1e-6  # Tolerance for root search (below this value = 0)
    de = 0.1  # Initial change in e for root search

    arr = []  # Array of eigenvalues
    cur_energy_looking_for = 0
    energy_guess = 0  # Initial energy guess, which will increase with a certain step size depending on the potential

    possible_potentials = [V_harmonic, V_triangular_well]  # Possible potential functions SO FAR
    folder_names = ["Harmonic_Oscillator_Images", "Finite_Triangular_Well_Images"]
    energy_steps = [0.1, 0.5]  # The energy step size for each potential in the possible_potentials[] list

    num_states_to_find = 4  # USER CAN EDIT THIS: how many states that the user wants to find
    chosen_potential_index = 1  # USER CAN EDIT THIS: which potential (index) from possible_potentials array

    # This loop keeps going until num_states_to_find states are found.
    # TODO: if one state is skipped by accident, this loop becomes infinite.
    while cur_energy_looking_for < num_states_to_find:

        # The final eigenvalue that the program guesses based on the energy_guess and potential.
        eigenvalue = secant(n_secant, tolerance, energy_guess, de, possible_potentials[chosen_potential_index])

        # Proceed only if secant() converged
        if eigenvalue is not None and not np.isnan(eigenvalue):

            # If the number of nodes is correct
            if num_nodes() == cur_energy_looking_for:
                print(f"The n={num_nodes()} eigenvalue: {eigenvalue}")
                arr.append(f"The n={num_nodes()} eigenvalue: {eigenvalue}")

                # Printing wavefunction values to a file in order to plot it
                x = x1
                mh = m*h
                with open("wavefunction_data.txt", 'w') as dataFile:
                    for j in np.arange(0, n_runge_kutta + 1, m):
                        dataFile.write(f"{x} {u[j]}\n")
                        x += mh

                # Plotting wavefunction from the file
                with open("wavefunction_data.txt", 'r') as dataFile:
                    nums = [[float(num) for num in line.split()] for line in dataFile.readlines()]

                    x_axis = np.array([point[0] for point in nums])
                    y_axis = np.array([point[1] for point in nums])

                    plt.plot(x_axis, y_axis)

                    plt.xlim([-6.0, 6.0])
                    plt.savefig(f"{folder_names[chosen_potential_index]}/n={num_nodes()}_wavefunction.png")
                    plt.show()

                # Now we are looking for the next energy level
                cur_energy_looking_for = cur_energy_looking_for + 1
        else:
            print(f"Non-convergence. Tested energy guess {energy_guess}.")

        # Increment the energy guess by the step size
        energy_guess = energy_guess + energy_steps[chosen_potential_index]

    print(arr)


if __name__ == '__main__':
    main()
