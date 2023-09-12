import torch, time, sys
import autograd
import autograd.numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.integrate

solve_ivp = scipy.integrate.solve_ivp

EXPERIMENT_DIR = './experiment-spring'
sys.path.append(EXPERIMENT_DIR)

from data import get_dataset, get_field, get_trajectory, dynamics_fn, hamiltonian_fn
from nn_models import MLP
from hnn import HNN
from utils import L2_loss


DPI = 300
FORMAT = 'pdf'
LINE_SEGMENTS = 10
ARROW_SCALE = 30
ARROW_WIDTH = 6e-3
LINE_WIDTH = 2
RK4 = ''

def get_args():
    return {'input_dim': 2,
         'hidden_dim': 200,
         'learn_rate': 1e-3,
         'nonlinearity': 'tanh',
         'total_steps': 10000,
         'field_type': 'solenoidal',
         'print_every': 200,
         'name': 'spring',
         'gridsize': 10,
         'input_noise': 0.5,
         'seed': 0,
         'save_dir': './{}'.format(EXPERIMENT_DIR),
         'fig_dir': './figures'}

class ObjectView(object):
    def __init__(self, d): self.__dict__ = d


def get_model(args, baseline):
    output_dim = args.input_dim if baseline else 2
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = HNN(args.input_dim, differentiable_model=nn_model,
                field_type=args.field_type, baseline=baseline)

    model_name = 'baseline' if baseline else 'hnn'
    path = "{}/spring{}-{}-{}.tar".format(args.save_dir, RK4, args.field_type, model_name)
    model.load_state_dict(torch.load(path))
    return model


def get_vector_field(model, **kwargs):
    field = get_field(**kwargs)
    np_mesh_x = field['x']

    # run model
    mesh_x = torch.tensor(np_mesh_x, requires_grad=True, dtype=torch.float32)
    mesh_dx = model.time_derivative(mesh_x)
    return mesh_dx.data.numpy()


def integrate_model(model, t_span, y0, **kwargs):
    def fun(t, np_x):
        x = torch.tensor(np_x, requires_grad=True, dtype=torch.float32).view(1, 2)
        dx = model.time_derivative(x).data.numpy().reshape(-1)
        return dx

    return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)


def integrate_models(x0=np.asarray([1, 0]), t_span=[0, 5], t_eval=None, noise_std=0.1):
    # integrate along ground truth vector field
    kwargs = {'t_eval': t_eval, 'rtol': 1e-12}
    true_path = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=x0, **kwargs)
    true_x = true_path['y'].T

    # rescale time to compensate for noise effects, as described in appendix
    t_span[1] *= 1 + .9 * noise_std
    t_eval *= 1 + .9 * noise_std

    # integrate along baseline vector field
    base_path = integrate_model(base_model, t_span, x0, **kwargs)
    base_x = base_path['y'].T
    _tmp = torch.tensor(true_x, requires_grad=True, dtype=torch.float32)

    # integrate along HNN vector field
    hnn_path = integrate_model(hnn_model, t_span, x0, **kwargs)
    hnn_x = hnn_path['y'].T
    _tmp = torch.tensor(true_x, requires_grad=True, dtype=torch.float32)

    return true_x, base_x, hnn_x


def energy_loss(true_x, integrated_x):
    true_energy = (true_x ** 2).sum(1)
    integration_energy = (integrated_x ** 2).sum(1)
    return np.mean((true_energy - integration_energy) ** 2)


if __name__ == '__main__':

    args = ObjectView(get_args())

    ###### PLOT ######
    fig = plt.figure(figsize=(11.3, 3.1), facecolor='white', dpi=DPI)

    # plot physical system
    fig.add_subplot(1, 4, 1, frameon=True)
    plt.xticks([]);
    plt.yticks([])
    schema = mpimg.imread(EXPERIMENT_DIR + '/mass-spring.png')
    plt.imshow(schema)
    plt.title("Mass-spring system", pad=10)

    # plot dynamics
    fig.add_subplot(1, 4, 2, frameon=True)
    x, y, _, _, _ = get_trajectory(radius=1, y0=np.array([1, 0]))
    N = len(x)
    point_colors = [(i / N, 0, 1 - i / N) for i in range(N)]
    plt.scatter(x, y, s=22, label='data', c=point_colors)

    # True underlying field determined from the values of dq/dt, dp/dt at each grid point
    field = get_field(gridsize=args.gridsize)
    plt.quiver(field['x'][:, 0], field['x'][:, 1], field['dx'][:, 0], field['dx'][:, 1],
               cmap='gray_r', scale=ARROW_SCALE, width=ARROW_WIDTH, color=(.2, .2, .2))

    plt.xlabel("$q$", fontsize=14)
    plt.ylabel("$p$", rotation=0, fontsize=14)
    plt.title("Data", pad=10)

    # plot baseline
    fig.add_subplot(1, 4, 3, frameon=True)

    t_span = [0, 30]
    y0 = np.asarray([1., 0])  # Initial point in phase space (q, p)
    kwargs = {'t_eval': np.linspace(t_span[0], t_span[1], 2000), 'rtol': 1e-12}

    base_model = get_model(args, baseline=True)
    base_field = get_vector_field(base_model, gridsize=args.gridsize)

    plt.quiver(field['x'][:, 0], field['x'][:, 1], base_field[:, 0], base_field[:, 1],
               cmap='gray_r', scale=ARROW_SCALE, width=ARROW_WIDTH, color=(.5, .5, .5))

    # The base/hnn model give the derivatives of x (q, p) wrt time, at any x
    # Use this information to plot the time evolution of (q, p)
    # Use Blue -> Red color gradient
    base_ivp = integrate_model(base_model, t_span, y0, **kwargs)
    for i, l in enumerate(np.split(base_ivp['y'].T, LINE_SEGMENTS)):
        color = (float(i) / LINE_SEGMENTS, 0, 1 - float(i) / LINE_SEGMENTS)
        plt.plot(l[:, 0], l[:, 1], color=color, linewidth=LINE_WIDTH)

    plt.xlabel("$q$", fontsize=14)
    plt.ylabel("$p$", rotation=0, fontsize=14)
    plt.title("Baseline NN", pad=10)

    # plot HNN
    fig.add_subplot(1, 4, 4, frameon=True)

    hnn_model = get_model(args, baseline=False)
    hnn_field = get_vector_field(hnn_model, gridsize=args.gridsize)

    plt.quiver(field['x'][:, 0], field['x'][:, 1], hnn_field[:, 0], hnn_field[:, 1],
               cmap='gray_r', scale=ARROW_SCALE, width=ARROW_WIDTH, color=(.5, .5, .5))

    hnn_ivp = integrate_model(hnn_model, t_span, y0, **kwargs)
    for i, l in enumerate(np.split(hnn_ivp['y'].T, LINE_SEGMENTS)):
        color = (float(i) / LINE_SEGMENTS, 0, 1 - float(i) / LINE_SEGMENTS)
        plt.plot(l[:, 0], l[:, 1], color=color, linewidth=LINE_WIDTH)

    plt.xlabel("$q$", fontsize=14)
    plt.ylabel("$p$", rotation=0, fontsize=14)
    plt.title("Hamiltonian NN", pad=10)

    plt.tight_layout();
    plt.show()

    # -----------------------------
    # QUANTITATIVE ANALYSIS
    # -----------------------------

    x0 = np.asarray([1, 0])

    # integration
    t_span = [0, 20]
    t_eval = np.linspace(t_span[0], t_span[1], 2000)
    true_x, base_x, hnn_x = integrate_models(x0=x0, t_span=t_span, t_eval=t_eval)

    # plotting
    tpad = 7

    fig = plt.figure(figsize=[12, 3], dpi=DPI)
    plt.subplot(1, 4, 1)
    plt.title("Predictions", pad=tpad)
    plt.xlabel('$q$')
    plt.ylabel('$p$')
    plt.plot(base_x[:, 0], base_x[:, 1], 'r-', label='Baseline NN', linewidth=2)
    plt.plot(hnn_x[:, 0], hnn_x[:, 1], 'b-', label='Hamiltonian NN', linewidth=2)
    plt.plot(true_x[:, 0], true_x[:, 1], 'k-', label='Ground truth', linewidth=2)
    plt.xlim(-1.2, 2)
    plt.ylim(-1.2, 2)
    plt.legend(fontsize=7)

    plt.subplot(1, 4, 2)
    plt.title("MSE between coordinates", pad=tpad)
    plt.xlabel('Time step')
    plt.plot(t_eval, ((true_x - base_x) ** 2).mean(-1), 'r-', label='Baseline NN', linewidth=2)
    plt.plot(t_eval, ((true_x - hnn_x) ** 2).mean(-1), 'b-', label='Hamiltonian NN', linewidth=2)
    plt.legend(fontsize=7)

    plt.subplot(1, 4, 3)
    plt.title("Total HNN-conserved quantity", pad=tpad)
    plt.xlabel('Time step')
    true_hq = hnn_model(torch.Tensor(true_x))[1].detach().numpy().squeeze()
    base_hq = hnn_model(torch.Tensor(base_x))[1].detach().numpy().squeeze()
    hnn_hq = hnn_model(torch.Tensor(hnn_x))[1].detach().numpy().squeeze()
    plt.plot(t_eval, true_hq, 'k-', label='Ground truth', linewidth=2)
    plt.plot(t_eval, base_hq, 'r-', label='Baseline NN', linewidth=2)
    plt.plot(t_eval, hnn_hq, 'b-', label='Hamiltonian NN', linewidth=2)
    plt.legend(fontsize=7)

    plt.subplot(1, 4, 4)
    plt.title("Total energy", pad=tpad)
    plt.xlabel('Time step')
    true_e = np.stack([hamiltonian_fn(c) for c in true_x])
    base_e = np.stack([hamiltonian_fn(c) for c in base_x])
    hnn_e = np.stack([hamiltonian_fn(c) for c in hnn_x])
    plt.plot(t_eval, true_e, 'k-', label='Ground truth', linewidth=2)
    plt.plot(t_eval, base_e, 'r-', label='Baseline NN', linewidth=2)
    plt.plot(t_eval, hnn_e, 'b-', label='Hamiltonian NN', linewidth=2)
    plt.legend(fontsize=7)

    plt.tight_layout();
    plt.show()
    fig.savefig('{}/spring-integration{}.{}'.format(args.fig_dir, RK4, FORMAT))