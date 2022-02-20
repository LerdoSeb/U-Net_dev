import math
# import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.style.use(['science'])
np.set_printoptions(precision=2)


def applyNoise(input_array, sigma, u_wall=10):
    output_array = np.random.normal(input_array, u_wall*sigma)
    return output_array


def plotFlowProfile(input_array, wall_height=20):

    if input_array.ndim == 2:
        t, h = input_array.shape

        v_step = wall_height / (h-1)
        v_steps = np.arange(0, wall_height + v_step, v_step).tolist()

        plt.plot(input_array[0], v_steps, label='t = 0')
        plt.plot(input_array[int(0.01*t)],
                 v_steps, label=f't = {int(0.01*t)}')
        plt.plot(input_array[int(0.05*t)],
                 v_steps, label=f't = {int(0.05*t)}')
        plt.plot(input_array[int(0.1*t)],
                 v_steps, label=f't = {int(0.1*t)}')
        plt.plot(input_array[int(0.25*t)],
                 v_steps, label=f't = {int(0.25*t)}')
        plt.plot(input_array[int(0.5*t)],
                 v_steps, label=f't = {int(0.5*t)}')
        plt.plot(input_array[int(0.75*t)],
                 v_steps, label=f't = {int(0.75*t)}')
        plt.plot(input_array[-1], v_steps, label=f't = {t}')

        plt.ylabel('Height $z$')
        plt.xlabel('Velocity $u$')
        plt.title('Flow Profile in Y-Direction ($U_{wall}$ in X-Direction)')
        plt.legend()
        plt.show()

    if input_array.ndim == 3:
        t, h, w = input_array.shape

        v_step = wall_height / (w-1)
        v_steps = np.arange(0, wall_height + v_step, v_step).tolist()

        # , sharex=True, sharey=True)
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        fig.suptitle(
            'Flow Profiles in X- and Y-Direction ($U_{wall}$ in X-Direction)', fontsize=10)

        ax1.set_title('Y-Direction', fontsize=8)
        ax1.set_ylabel('Height $y$')

        ax1.plot(input_array[0][:, int(h/2)], v_steps, label='t = 0')
        ax1.plot(input_array[int(0.01*t)][:, int(h/2)],
                 v_steps, label=f't = {int(0.01*t)}')
        ax1.plot(input_array[int(0.05*t)][:, int(h/2)],
                 v_steps, label=f't = {int(0.05*t)}')
        ax1.plot(input_array[int(0.1*t)][:, int(h/2)],
                 v_steps, label=f't = {int(0.1*t)}')
        ax1.plot(input_array[int(0.25*t)][:, int(h/2)],
                 v_steps, label=f't = {int(0.25*t)}')
        ax1.plot(input_array[int(0.5*t)][:, int(h/2)],
                 v_steps, label=f't = {int(0.5*t)}')
        ax1.plot(input_array[int(0.75*t)][:, int(h/2)],
                 v_steps, label=f't = {int(0.75*t)}')
        ax1.plot(input_array[-1][:, int(h/2)], v_steps, label=f't = {t}')

        ax2.set_title('X-Direction', fontsize=8)
        ax2.set_ylabel('Length $x$')

        ax2.plot(input_array[0, int(h/2), :], v_steps, label='t = 0')
        print('test length values')
        print(input_array[10, int(h/2), :])
        ax2.plot(input_array[int(0.01*t), int(h/2), :],
                 v_steps, label=f't = {int(0.01*t)}')
        ax2.plot(input_array[int(0.05*t), int(h/2), :],
                 v_steps, label=f't = {int(0.05*t)}')
        ax2.plot(input_array[int(0.1*t), int(h/2), :],
                 v_steps, label=f't = {int(0.1*t)}')
        ax2.plot(input_array[int(0.25*t), int(h/2), :],
                 v_steps, label=f't = {int(0.25*t)}')
        ax2.plot(input_array[int(0.5*t), int(h/2), :],
                 v_steps, label=f't = {int(0.5*t)}')
        ax2.plot(input_array[int(0.75*t), int(h/2), :],
                 v_steps, label=f't = {int(0.75*t)}')
        ax2.plot(input_array[-1, int(h/2), :], v_steps, label=f't = {t}')

        plt.xlabel('Velocity $u$')
        plt.legend()
        plt.show()

    if input_array.ndim == 4:
        t, d, h, w = input_array.shape

        v_step = wall_height / (w-1)
        v_steps = np.arange(0, wall_height + v_step, v_step).tolist()

        # , sharex=True, sharey=True)
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
        fig.suptitle(
            'Flow Profiles in X-,Y- and Z-Direction ($U_{wall}$ in X-Direction)', fontsize=10)

        ax1.set_title('Y-Direction', fontsize=8)
        ax1.set_ylabel('Height $y$')

        ax1.plot(input_array[0, int(h/2), :,  int(h/2)],
                 v_steps, label='t = 0')
        ax1.plot(input_array[int(0.01*t), int(h/2), :, int(h/2)],
                 v_steps, label=f't = {int(0.01*t)}')
        ax1.plot(input_array[int(0.05*t), int(h/2), :, int(h/2)],
                 v_steps, label=f't = {int(0.05*t)}')
        ax1.plot(input_array[int(0.1*t), int(h/2), :, int(h/2)],
                 v_steps, label=f't = {int(0.1*t)}')
        ax1.plot(input_array[int(0.25*t), int(h/2), :, int(h/2)],
                 v_steps, label=f't = {int(0.25*t)}')
        ax1.plot(input_array[int(0.5*t), int(h/2), :, int(h/2)],
                 v_steps, label=f't = {int(0.5*t)}')
        ax1.plot(input_array[int(0.75*t), int(h/2), :, int(h/2)],
                 v_steps, label=f't = {int(0.75*t)}')
        ax1.plot(input_array[-1,  int(h/2), :,  int(h/2)],
                 v_steps, label=f't = {t}')

        ax2.set_title('X-Direction', fontsize=8)
        ax2.set_ylabel('Length $x$')

        ax2.plot(input_array[0, int(h/2), int(h/2), :], v_steps, label='t = 0')
        print('test length values')
        print(input_array[10, int(h/2), int(h/2), :])
        ax2.plot(input_array[int(0.01*t), int(h/2), int(h/2), :],
                 v_steps, label=f't = {int(0.01*t)}')
        ax2.plot(input_array[int(0.05*t), int(h/2), int(h/2), :],
                 v_steps, label=f't = {int(0.05*t)}')
        ax2.plot(input_array[int(0.1*t), int(h/2), int(h/2), :],
                 v_steps, label=f't = {int(0.1*t)}')
        ax2.plot(input_array[int(0.25*t), int(h/2), int(h/2), :],
                 v_steps, label=f't = {int(0.25*t)}')
        ax2.plot(input_array[int(0.5*t), int(h/2), int(h/2), :],
                 v_steps, label=f't = {int(0.5*t)}')
        ax2.plot(input_array[int(0.75*t), int(h/2), int(h/2), :],
                 v_steps, label=f't = {int(0.75*t)}')
        ax2.plot(input_array[-1, int(h/2), int(h/2), :],
                 v_steps, label=f't = {t}')

        ax3.set_title('Z-Direction', fontsize=8)
        ax3.set_ylabel('Depth $z$')

        ax3.plot(input_array[0, :, int(h/2), int(h/2)], v_steps, label='t = 0')
        print('test depth values')
        print(input_array[10, :, int(h/2), int(h/2)])
        ax3.plot(input_array[int(0.01*t), :, int(h/2), int(h/2)],
                 v_steps, label=f't = {int(0.01*t)}')
        ax3.plot(input_array[int(0.05*t), :, int(h/2), int(h/2)],
                 v_steps, label=f't = {int(0.05*t)}')
        ax3.plot(input_array[int(0.1*t), :, int(h/2), int(h/2)],
                 v_steps, label=f't = {int(0.1*t)}')
        ax3.plot(input_array[int(0.25*t), :, int(h/2), int(h/2)],
                 v_steps, label=f't = {int(0.25*t)}')
        ax3.plot(input_array[int(0.5*t), :, int(h/2), int(h/2)],
                 v_steps, label=f't = {int(0.5*t)}')
        ax3.plot(input_array[int(0.75*t), :, int(h/2), int(h/2)],
                 v_steps, label=f't = {int(0.75*t)}')
        ax3.plot(input_array[-1, :, int(h/2), int(h/2)],
                 v_steps, label=f't = {t}')

        plt.xlabel('Velocity $u$')
        plt.legend()
        plt.show()


def compareFlowProfile(prediction_array, target_array, wall_height=20, u_wall=10, sigma=0.3):
    t, h, w = prediction_array.shape
    v_step = wall_height / (w-1)
    v_steps = np.arange(0, wall_height + v_step, v_step).tolist()

    for i in range(t):
        fig, ax = plt.subplots()
        ax.plot(v_steps, prediction_array[i][:, int(
            w/2)], label='predicted profile')
        ax.plot(v_steps, target_array[i][:, int(
            w/2)], label='noisy analytical profile')
        plt.yticks(range(int(-u_wall), int(u_wall*(2)+1), 5))
        plt.xlabel('Height $z$')
        plt.ylabel('Velocity $u$')
        plt.title('The Startup Couette Problem')
        plt.legend()
        # fig.set_size_inches(5, 5)
        plt.show()
        fig.savefig(f'pred_vs_noisy_target_3e-1_{i}.svg')


def plotVelocityField(input_array, wall_height=20):
    if input_array.ndim == 2:
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

        plt.ylabel('Height $z$')
        plt.xlabel('Depth $x$')
        plt.title('The Startup Couette Velocity Field (Cross-Section)')
        plt.show()

    if input_array.ndim == 3:
        d, h, w = input_array.shape
        v_step = wall_height / (h-1)
        v_steps = np.arange(0, wall_height + v_step, v_step).tolist()

        X1, Y1 = np.meshgrid(v_steps, v_steps)
        u = input_array[int(0.5*d)]
        v = np.zeros(shape=u.shape)

        # set color field for better visualisation
        n = -2
        color = np.sqrt(((v-n)/2)*2 + ((u-n)/2)*2)

        # set plot parameters
        # u - velocity component in x-direction
        # v - velocity component in y-direction
        fig, ax = plt.subplots()
        ax.quiver(X1, Y1, u, v, color, alpha=0.75)

        plt.ylabel('Height $z$')
        plt.xlabel('Depth $x$')
        plt.title('The Startup Couette Velocity Field (Cross-Section)')
        plt.show()


def compareVelocityField(prediction_array, target_array, wall_height=20):
    t, h, w = prediction_array.shape
    v_step = wall_height / (h-1)
    v_steps = np.arange(0, wall_height + v_step, v_step).tolist()

    X1, Y1 = np.meshgrid(v_steps, v_steps)

    for i in range(t):
        # set plot parameters
        # u - velocity component in x-direction
        # v - velocity component in y-direction
        u_pred = prediction_array[i]
        u_targ = target_array[i]
        v = np.zeros(shape=u_pred.shape)

        # set color field for better visualisation
        n = -2
        color = np.sqrt(((v-n)/2)*2 + ((u_pred-n)/2)*2)

        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.quiver(X1, Y1, u_pred, v, color, alpha=0.75)
        ax1.set_aspect('equal')
        ax1.set_title('predicted')
        ax1.set_xlabel('Depth $x$')
        ax1.set_ylabel('Height $z$')

        ax2.quiver(X1, Y1, u_targ, v, color, alpha=0.75)
        ax2.set_aspect('equal')
        ax2.set_title('noisy analytical')
        ax2.set_xlabel('Depth $x$')
        fig.suptitle('The Startup Couette Velocity Field (Cross-Section)')
        fig.set_size_inches(4, 4)
        plt.show()
        fig.savefig(f'pred_vs_noisy_target_v_field_3e-1_{i}.svg')


def expandVector2Matrix(input_list):
    output_list = []
    N = input_list[0].shape[0]
    print('dimensions of array in list: ', N)

    for element in input_list:
        dummy = (np.vstack([element]*N)).T
        output_list.append(dummy)
    return output_list


def expandMatrix2Tensor(input_list):
    output_list = []
    h, w = input_list[0].shape

    for element in input_list:
        dummy = np.stack(([element]*h))
        output_list.append(dummy)

    return output_list


def list2array(input_list):
    output_array = np.stack(input_list)
    # output_array = np.rollaxis(output_array, -1)
    # Sanity check: data type and dimensions
    # print(f'Data type of my_2d_array: {type(my_2d_array)}')
    # print(f'Shape of my_2d_array: {my_2d_array.shape}')

    return output_array


def couetteSolver(desired_timesteps, u_wall=10, wall_height=20, nu=2, vertical_resolution=63, sigma=0):
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
    my_data.append(np.array(my_time_zero))
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
        my_data.append(np.flip(np.array(dummy_vector)))

    # Sanity check: Do the lengths of the lists make sense?
    # print('length of my_data: {length}'.format(length=len(my_data)))
    # print('length of element: {length}'.format(length=len(my_data[0])))

    # Sanity check: Do the first and last elements in the list make sense?
    # print('couette solver shape at [0]: ', my_data[0].shape)
    # print('couette solver shape at [10]: ', my_data[10].shape)
    # print('couette solver element at [10]: ', my_data[10])
    # print('couette solver type of list element [0]: ', type(my_data[0]))
    # print('couette solver type of list element [5]: ', type(my_data[5]))
    # print(my_data[desired_timesteps-1])

    return my_data


def my1DCouetteSolver(desired_timesteps, u_wall=10, wall_height=20, nu=2, vertical_resolution=63, sigma=0):

    my_data = couetteSolver(desired_timesteps, u_wall,
                            wall_height, nu, vertical_resolution, sigma)
    my_1D_array = list2array(my_data)

    if sigma != 0:
        my_1D_array = applyNoise(my_1D_array, (sigma*u_wall))

    return my_1D_array


def my2DCouetteSolver(desired_timesteps, u_wall=10, wall_height=20, nu=2, vertical_resolution=63, sigma=0):

    my_1d_list = couetteSolver(
        desired_timesteps, u_wall, wall_height, nu, vertical_resolution)

    my_2d_list = expandVector2Matrix(my_1d_list)
    # print(type(my_2d_list[0]))

    my_2d_array = list2array(my_2d_list)

    if sigma != 0:
        my_2d_array = applyNoise(my_2d_array, sigma)

    # Sanity check: Plot the flow data
    # plotFlowProfile(my_2d_array, wall_height, vertical_resolution)

    return my_2d_array


def my3DCouetteSolver(desired_timesteps, u_wall=10, wall_height=20, nu=2, vertical_resolution=63, sigma=0):

    my_1d_list = couetteSolver(
        desired_timesteps, u_wall, wall_height, nu, vertical_resolution)

    my_2d_list = expandVector2Matrix(my_1d_list)

    my_3d_list = expandMatrix2Tensor(my_2d_list)

    my_3d_array = list2array(my_3d_list)

    if sigma != 0:
        my_3d_array = applyNoise(my_3d_array, sigma)

    # Sanity check: Plot the flow data
    # plotFlowProfile(my_2d_array, wall_height, vertical_resolution)

    return my_3d_array


if __name__ == '__main__':
    t = 100
    v_res = 31

    # my_data = couetteSolver(desired_timesteps=t, vertical_resolution=v_res, sigma=0)
    # print(f'len of couette solver list = {len(my_data)}')
    # print(f'sample of couette solver list = {my_data[50][-1]}')
    # plotFlowProfile(my_data)

    # my_data = my1DCouetteSolver(desired_timesteps=t, vertical_resolution=v_res, sigma=0)
    # print(f'shape of 1d array = {my_data.shape}')
    # print(f'sample of 1d array = {my_data[50]}')
    # plotFlowProfile(my_data)

    # my_data = my2DCouetteSolver(desired_timesteps=t, vertical_resolution=v_res, sigma=0)
    # print(f'shape of 2d array = {my_data.shape}')
    # print(f'sample of 2d array = {my_data[50,:, 16]}')
    # plotFlowProfile(my_data)
    # plotVelocityField(my_data[8])

    # my_data = my3DCouetteSolver(desired_timesteps=t, vertical_resolution=v_res, sigma=0.1)
    # print(f'shape of 3d array = {my_data.shape}')
    # plotFlowProfile(my_data)
