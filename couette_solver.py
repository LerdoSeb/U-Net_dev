import math
# import torch
import numpy as np
# import matplotlib.pyplot as plt


def my2DCouetteSolver():
    my_data = []  # Container for all time step vectors
    u_wall = 10  # make user input
    wall_height = 20  # make user input
    nu = 2  # make user input
    vertical_resolution = 64  # make user input
    my_duration = 1100  # DONT MAKE USER Input
    # The relaxation time is roughly: t = h*h/nu. We want 1000 time samples before
    # start up. Therefor we need the general formula to create the list of timesteps:
    # 1000*stepsize = h*h/nu
    # 1100*stepsize = 1.1 * h*h/nu
    # stepsize = (1/1000)* h*h/nu
    # upperbound = 1100*step
    my_timestep = (1/1000)*(wall_height**2)/(nu)
    my_time_upperbound = (1100)*my_timestep
    my_timesteps = np.arange(
        my_timestep, my_time_upperbound, my_timestep).tolist()
    my_time_zero = [0]*65
    # print('length of time_zero: {length}'.format(length=len(my_time_zero)))
    my_data.append(my_time_zero)
    # For the initial u_net test cases, a picture-like resolution of 64x64 is
    # desired. Therefor we need the general formula to create the list of vertical
    # steps. Caution, these must include h=0 and h=wall_height:
    # stepsize = height/resolution
    # upperbound = height + stepsize
    my_vertical_step = wall_height / vertical_resolution
    my_vertical_upperbound = wall_height + my_vertical_step
    my_vertical_steps = np.arange(
        my_vertical_step, my_vertical_upperbound, my_vertical_step).tolist()

    for t in my_timesteps:
        dummy_vector = []
        dummy_vector.append(0.0)
        for y in my_vertical_steps:
            result = 2*u_wall/(math.pi)
            sum = 0
            for n in list(range(1, 31)):
                sum += ((-1**n)/n) * math.exp(-(n**2)*(math.pi**2)*nu*t
                                              / (wall_height**2)) * math.sin(n*math.pi*(1-(y/wall_height)))
            result *= sum
            result += u_wall*(y/wall_height)
            dummy_vector.append(result)
        my_data.append(dummy_vector)

    # Sanity check: Do the lengths of the lists make sense?
    # print('length of my_data: {length}'.format(length=len(my_data)))
    # print('length of element: {length}'.format(length=len(my_data[0])))

    # Sanity check: Do the first and last elements in the list make sense?
    # print(my_data[0])
    # print(my_data[1090])

    my_2d_data = []
    N = vertical_resolution
    for element in my_data:
        dummy = (np.vstack([np.flip(element)]*N)).T
        my_2d_data.append(dummy)

    # print(my_2d_data)
    # print('I am here.')

    # Sanity check: Do all of the elements in the lists make sense? Plot samples of
    # the list against their respective heights

    # my_vertical_steps_with_zero = np.arange(0, my_vertical_upperbound, my_vertical_step).tolist()
    # y_plot = my_vertical_steps_with_zero
    # plt.plot(my_data[0], y_plot, label='y = 0')
    # plt.plot(my_data[10], y_plot, label='y = 10')
    # plt.plot(my_data[25], y_plot, label='y = 25')
    # plt.plot(my_data[50], y_plot, label='y = 50')
    # plt.plot(my_data[100], y_plot, label='y = 100')
    # plt.plot(my_data[250], y_plot, label='y = 250')
    # plt.plot(my_data[500], y_plot, label='y = 500')
    # plt.plot(my_data[750], y_plot)
    # plt.plot(my_data[1000], y_plot)
    # plt.ylabel('Height y')
    # plt.xlabel('Velocity u')
    # plt.title('The Startup Couette Problem')
    # plt.legend()
    # plt.show()

    # Now plot the data with 0-mean, 50% standard deviation gaussian white noise:

    # my_data_array = np.array(my_data)
    # my_data_noisy = my_data_array + np.random.normal(0, 0.5, my_data_array.shape)

    # my_vertical_steps_with_zero = np.arange(
    #    0, my_vertical_upperbound, my_vertical_step).tolist()
    # y_plot = my_vertical_steps_with_zero

    # plt.plot(my_data_noisy[0], y_plot, label='y = 0')
    # plt.plot(my_data_noisy[10], y_plot, label='y = 10')
    # plt.plot(my_data_noisy[25], y_plot, label='y = 25')
    # plt.plot(my_data_noisy[50], y_plot, label='y = 50')
    # plt.plot(my_data_noisy[100], y_plot, label='y = 100')
    # plt.plot(my_data_noisy[250], y_plot, label='y = 250')
    # plt.plot(my_data_noisy[500], y_plot, label='y = 500')
    # plt.plot(my_data[750], y_plot)
    # plt.plot(my_data[1000], y_plot)

    # plt.ylabel('Height y')
    # plt.xlabel('Velocity u')
    # plt.title('The Startup Couette Problem (Noisy)')
    # plt.legend()
    # plt.show()
    # print(len(my_2d_data))
    return my_2d_data


if __name__ == '__main__':
    my_data = my2DCouetteSolver()
    print(my_data[0].shape)
    print(my_data[10].shape)
