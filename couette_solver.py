import math
# import torch
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['science'])
np.set_printoptions(precision=2)


def applyNoise(input_array, sigma):
    output_array = np.random.normal(input_array, abs(input_array*sigma))
    return output_array


def plotFlowProfile(input_array, wall_height, vertical_resolution):
    t, h, w = input_array.shape
    v_step = wall_height / vertical_resolution
    v_steps = np.arange(0, wall_height + v_step, v_step).tolist()

    plt.plot(input_array[0][:, int(h/2)], v_steps, label='t = 0')
    plt.plot(input_array[int(0.01*t)][:, int(h/2)],
             v_steps, label=f't = {int(0.01*t)}')
    plt.plot(input_array[int(0.05*t)][:, int(h/2)],
             v_steps, label=f't = {int(0.05*t)}')
    plt.plot(input_array[int(0.1*t)][:, int(h/2)],
             v_steps, label=f't = {int(0.1*t)}')
    plt.plot(input_array[int(0.25*t)][:, int(h/2)],
             v_steps, label=f't = {int(0.25*t)}')
    plt.plot(input_array[int(0.5*t)][:, int(h/2)],
             v_steps, label=f't = {int(0.5*t)}')
    plt.plot(input_array[int(0.75*t)][:, int(h/2)],
             v_steps, label=f't = {int(0.75*t)}')
    plt.plot(input_array[-1][:, int(h/2)], v_steps, label=f't = {t}')

    plt.ylabel('Height $y$')
    plt.xlabel('Velocity $u$')
    plt.title('The Startup Couette Problem')
    plt.legend()
    plt.show()


def plotVelocityField(input_array, wall_height=20):
    h, w = input_array.shape
    v_step = wall_height / (h-1)
    v_steps = np.arange(0, wall_height + v_step, v_step).tolist()

    X1, Y1 = np.meshgrid(v_steps, v_steps)
    u = input_array
    v = np.zeros(shape=u.shape)

    # set color field for better visualisation
    n = -2
    color = np.sqrt(((v-n)/2)*2 + ((u-n)/2)*2)

    # set plot parameters
    # u - velocity component in x-direction
    # v - velocity component in y-direction
    fig, ax = plt.subplots()
    ax.quiver(X1, Y1, u, v, color, alpha=0.75)

    plt.ylabel('Height $y$')
    plt.xlabel('Depth $x$')
    plt.title('The Startup Couette Velocity Field (Cross-Section)')
    plt.show()


def expandVector2Matrix(input_list):
    output_list = []
    N = len(input_list[0])

    for element in input_list:
        dummy = (np.vstack([np.flip(element)]*N)).T
        output_list.append(dummy)
    return output_list


def list2array(input_list):
    output_array = np.dstack(input_list)
    output_array = np.rollaxis(output_array, -1)
    # Sanity check: data type and dimensions
    # print(f'Data type of my_2d_array: {type(my_2d_array)}')
    # print(f'Shape of my_2d_array: {my_2d_array.shape}')

    return output_array


def my2DCouetteSolver(desired_timesteps, u_wall=10, wall_height=20, nu=2, vertical_resolution=63, sigma=0):
    my_data = []  # Container for all time step vectors

    # The relaxation time is roughly: t = h*h/nu. We want 'desired_timesteps' of
    # time samples before start up. Therefor we need the general formula to
    # create the list of timesteps:
    # relax_time = h*h/nu = upperbound
    # desired_timesteps*stepsize = h*h/nu
    # stepsize = (1/desired_timesteps)* h*h/nu
    my_timestep = (1/desired_timesteps)*(wall_height**2)/(nu)
    my_time_upperbound = (wall_height**2)/(nu)
    my_timesteps = np.arange(
        my_timestep, my_time_upperbound, my_timestep).tolist()
    my_time_zero = [0.0]*(vertical_resolution+1)
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
    # print(my_data[desired_timesteps-1])

    my_2d_list = expandVector2Matrix(my_data)

    my_2d_array = list2array(my_2d_list)

    if sigma != 0:
        my_2d_array = applyNoise(my_2d_array, sigma)

    # Sanity check: Plot the flow data
    plotFlowProfile(my_2d_array, wall_height, vertical_resolution)

    return my_2d_array


if __name__ == '__main__':
    t = 1000
    v_res = 64
    sigma = 0.3
    my_data = my2DCouetteSolver(
        desired_timesteps=t, vertical_resolution=v_res, sigma=sigma)

    plotVelocityField(my_data[128])
