import numpy as np
import math
import matplotlib.pyplot as plt

n = 400  # Number of iterations for Runge-Kutta
ni = 10  # Number of iterations for secant
m = 1  # Every m indices, output of wavefunction is printed to the file
x1 = -10.0
x2 = 10.0
h = (x2-x1) / n  # To have n steps, step size in x must be h
ul0 = np.empty(n + 1)  # phi integrated from the left to the turning point
ul1 = np.empty(n + 1)  # first derivative of phi integrated from the left to the turning point
ur0 = np.empty(n + 1)  # phi integrated from the right to the turning point
ur1 = np.empty(n + 1)  # first derivative of phi integrated from the right to the turning point
ul = np.empty(2)  # values of ul0[] and ul1[] at a particular point
ur = np.empty(2)  # values of ul0[] and ul1[] at a particular point
u = np.empty(n + 1)  # The actual wavefunction phi, using ul for the left and ur for the right
countPrint = 0


# Calculates the quantity needed for secant to minimize
def f(e):
    # These are the global variables that will be edited
    global countPrint
    global ul0, ul1, ur0, ur1, ul, ur, u

    # Initial values
    ul0[0] = ul[0] = 0  # Left phi starts at 0
    ul1[0] = ul[1] = 0.01  # Left phi derivative starts at 0.01

    ur0[0] = ur[0] = 0  # Right phi starts at 0
    ur1[0] = ur[1] = -0.01  # Right phi derivative starts at -0.01

    turnIndex = turning_index(e)
    # turnIndex = 200
    if turnIndex == 0:
        turnIndex = 2

    if countPrint == 0:
        print(f"index: {turnIndex}, x-val: {x1 + (turnIndex * h)}")
        print(f"index 0 to {turnIndex+1}")
        print(f"index {n} to {turnIndex-1}")
        print(f"{len(ul0)} {len(ur0)}")
        countPrint = countPrint + 1

    # Runge Kutta for ul, from -5 -> turning point + 1
    for i in range(turnIndex + 2):
        x = x1 + (i * h)
        ul = rungeKutta(ul, x, e, h)
        ul0[i+1] = ul[0]
        ul1[i+1] = ul[1]

    # Runge Kutta for ur, from 10 -> turning point - 1
    for i in np.arange(n, turnIndex - 2, -1):
        if i >= 0:
            x = x2 - ((n - i) * h)
            ur = rungeKutta(ur, x, e, -h)
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

    return (ul0[turnIndex-1]-ul0[turnIndex+1]-ur0[turnIndex-1]+ur0[turnIndex+1]) / (2*h*ur0[turnIndex])


# Performs the root-searching algorithm and returns if there is a root, within a given tolerance
def secant(n, tolerance, x, dx):
    k = 0
    x1 = x + dx
    while math.fabs(dx) > tolerance and k < n:
        d = f(x1) - f(x)
        x2 = x1 - f(x1) * (x1 - x) / d
        x = x1
        x1 = x2
        dx = x1 - x
        k = k + 1

    if k == n:
        print(f"Convergence not found after {n} iterations")

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
# u is the 2-component numpy array that represents the wavefunction
# x is the x-value
# e is the energy eigenvalue guess
# dx is the step in x
def rungeKutta(u, x, e, dx):
    c1 = g(u, x, e)

    c2 = u + (dx * c1/2)
    c2 = g(c2, x + (dx/2), e)

    c3 = u + (dx * c2/2)
    c3 = g(c3, x + (dx/2), e)

    c4 = u + (dx * c3)
    c4 = g(c4, x + dx, e)

    return u + (dx * ((c1 + (2 * (c2 + c3)) + c4) / 6))


# Generalized velocity vector
# y is a 2-component numpy array that contains the two dynamical variables, psi and psi prime in this case.
# x is the x-value.
# e is the energy eigenvalue guess.
def g(y, x, e):
    return np.array([y[1], (-2 * y[0]) * (e - V(x))])


# Potential function (Harmonic oscillator)
def V(x):
    return 0.5 * math.pow(x, 2)


# Calculates turning index using potential V(x)
def turning_index(e):
    for i in range(n):
        xr = x1 + (i * h)
        if xr > 0.0 and (e - V(xr)) <= 0.1:
            return i

    return 0


def main():
    tolerance = 1e-6  # Tolerance for root search (below this value = 0)
    e = 5.5  # Initial guess for eigenenergy
    de = 0.1  # Initial change in e for root search

    """
    The root search. Returns the eigenvalue. The function of the root search depends on phi left anf phi right.
    phi left and phi right depend on the Runge-Kutta algorithm to create them.
    The Runge-Kutta depends on the generalized velocity vector g().
    g() depends on the second derivative of phi, which depends on the eigenenergy.
    Everything is called through secant(). 
    """

    for i, energy_guess in enumerate(np.arange(0.5, 7.5, 1)):

        eigenvalue = secant(n, tolerance, energy_guess, de)
        x = x1
        mh = m*h

        with open("wavefunction_data.txt", 'w') as dataFile:
            for j in np.arange(0, n+1, m):
                dataFile.write(f"{x} {u[j]}\n")
                x += mh

        print(f"The eigenvalue: {eigenvalue}")

        # Plotting wavefunction
        with open("wavefunction_data.txt", 'r') as dataFile:
            nums = [[float(num) for num in line.split()] for line in dataFile.readlines()]

            x_axis = np.array([point[0] for point in nums])
            y_axis = np.array([point[1] for point in nums])

            plt.plot(x_axis, y_axis)

            plt.xlim([-6.0, 6.0])
            plt.savefig(f"images/n={i}_wavefunction.png")
            plt.show()


if __name__ == '__main__':
    main()
