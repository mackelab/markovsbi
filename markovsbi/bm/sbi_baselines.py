import sbi
from sbi.inference import SNLE, SNRE, SNPE, NPSE


from markovsbi.bm.api_utils import NPEModel, NPESummaryModel, NSEModel, SBIModel
from markovsbi.utils.prior_utils import Normal

import torch
import numpy as np
import torch.distributions as dist


def run_factorized_nle_or_nre(cfg, task, data, method="nle"):
    if method == "nle":
        method = SNLE
    elif method == "nre":
        method = SNRE
    else:
        raise ValueError(f"Unknown method {method}")

    thetas = data["thetas"]
    xs = data["xs"]
    T = int(xs.shape[1])
    xs = xs.reshape(xs.shape[0], -1)

    # TODO: This would be the correct think to do
    # NOTE: Both is fine!
    # # Flatten xs
    # T = int(xs.shape[1])
    # # Assume xs is markov we only need to model density of last xs
    # cond_xs = xs[:, :-1, :]
    # cond_xs = cond_xs.reshape(-1, cond_xs.shape[-1])

    # # We have to extend thetas to also get cond_xs
    # thetas = torch.concatenate([thetas, cond_xs], dim=-1)
    # xs = xs[:, -1, :]

    # Setup inference
    prior = task.get_prior()
    prior_torch = convert_prior_to_torch(prior)

    init_params = dict(cfg.method.params_init)
    inf = method(prior=prior_torch, **init_params)

    # Perform training
    inf = inf.append_simulations(thetas, xs)
    train_kwargs = dict(cfg.method.params_train)
    _ = inf.train(**train_kwargs)

    # Build posterior
    posterior_kwargs = dict(cfg.method.params_build_posterior)
    posterior = inf.build_posterior(**posterior_kwargs)

    return SBIModel(posterior, T, cfg=cfg)


class TimeSeriesEmbeddingNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=2):
        super(TimeSeriesEmbeddingNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gru = torch.nn.GRU(
            self.input_dim, self.output_dim, num_layers=num_layers, batch_first=True
        )

    def forward(self, x):
        *batch_shape, n_samples, n_dim = x.shape
        x = x.view(-1, n_samples, n_dim)
        # Replace nans with zeros
        mask = torch.isnan(x).any(-1, keepdim=True).to(torch.int32)
        indices = torch.argmax(mask, dim=1).squeeze(1) - 1
        x = torch.nan_to_num(x, nan=0.0)
        hs = self.gru(x)[0]
        x = hs[torch.arange(hs.shape[0]), indices, :]
        x = x.view(*batch_shape, self.output_dim)
        return x


from sbi.neural_nets.embedding_nets import PermutationInvariantEmbedding


class IncrementEmbeddingNet(PermutationInvariantEmbedding):
    def __init__(self, input_dim, trial_dim, output_dim, num_layers=2):
        trial_net = torch.nn.Sequential(
            torch.nn.Linear(input_dim * 2, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, trial_dim),
        )
        super(IncrementEmbeddingNet, self).__init__(
            trial_net, trial_dim, output_dim=output_dim
        )
        self.output_dim = output_dim

    def forward(self, x):
        *batch_shape, n_samples, n_dim = x.shape
        x = x.view(-1, n_samples, n_dim)
        # Get window of data (x0,x1), (x1,x2), (x2,x3), ...
        x = torch.cat([x[:, :-1], x[:, 1:]], dim=-1)
        out = super(IncrementEmbeddingNet, self).forward(x)
        return out.view(*batch_shape, self.output_dim)


def run_npe_embedding_network(cfg, task, data, rng_method):
    method = SNPE

    # Data
    thetas = data["thetas"]
    xs = data["xs"]

    # Data augmentation for different sequence lengths
    T_max = xs.shape[1]
    for t in range(2, T_max):
        idx = torch.randint(
            0,
            xs.shape[0],
            (
                int(
                    cfg.method.subseq_data_augmentation_fraction
                    * cfg.task.num_simulations
                    / (T_max - 1)
                ),
            ),
        )
        xs_subset = xs[idx]
        theta_subset = thetas[idx]
        xs_subset[:, t:] = torch.nan  # Cut off data after t
        xs = torch.cat([xs, xs_subset], dim=0)
        thetas = torch.cat([thetas, theta_subset], dim=0)

    # Setup inference
    prior = task.get_prior()
    prior_torch = convert_prior_to_torch(prior)

    neural_net_params = cfg.method.neural_net

    if neural_net_params.name == "rnn":
        embedding_net = TimeSeriesEmbeddingNet(
            input_dim=xs.shape[-1],
            output_dim=neural_net_params.output_dim,
            num_layers=neural_net_params.num_layers,
        )
        density_estimator = sbi.utils.posterior_nn(
            model=cfg.method.params_init.density_estimator,
            embedding_net=embedding_net,
            z_score_x="none",
        )
    elif neural_net_params.name == "increment":
        embedding_net = IncrementEmbeddingNet(
            input_dim=xs.shape[-1],
            trial_dim=neural_net_params.trial_dim,
            output_dim=neural_net_params.output_dim,
        )
        density_estimator = sbi.utils.posterior_nn(
            model=cfg.method.params_init.density_estimator,
            embedding_net=embedding_net,
            z_score_x="none",
        )
    else:
        raise ValueError(f"Unknown neural net {neural_net_params.name}")

    inf = method(prior=prior_torch, density_estimator=density_estimator)

    # Perform training
    inf = inf.append_simulations(thetas, xs, exclude_invalid_x=False)
    train_kwargs = dict(cfg.method.params_train)
    _ = inf.train(**train_kwargs)

    # Build posterior
    posterior_kwargs = dict(cfg.method.params_build_posterior)
    posterior = inf.build_posterior(**posterior_kwargs)

    return NPEModel(posterior, cfg=cfg)


def run_nse_embedding_network(cfg, task, data, rng_method):
    method = NPSE

    # Data
    thetas = data["thetas"]
    xs = data["xs"]

    # Data augmentation for different sequence lengths
    T_max = xs.shape[1]
    for t in range(2, T_max):
        idx = torch.randint(
            0,
            xs.shape[0],
            (
                int(
                    cfg.method.subseq_data_augmentation_fraction
                    * cfg.task.num_simulations
                    / (T_max - 1)
                ),
            ),
        )
        xs_subset = xs[idx]
        theta_subset = thetas[idx]
        xs_subset[:, t:] = torch.nan  # Cut off data after t
        xs = torch.cat([xs, xs_subset], dim=0)
        thetas = torch.cat([thetas, theta_subset], dim=0)

    # Setup inference
    prior = task.get_prior()
    prior_torch = convert_prior_to_torch(prior)

    neural_net_params = cfg.method.neural_net

    if neural_net_params.name == "rnn":
        embedding_net = TimeSeriesEmbeddingNet(
            input_dim=xs.shape[-1],
            output_dim=neural_net_params.output_dim,
            num_layers=neural_net_params.num_layers,
        )
        score_estimator = sbi.neural_nets.posterior_score_nn(
            sde_type="vp",
            embedding_net=embedding_net,
            z_score_x="none",
        )
    elif neural_net_params.name == "increment":
        embedding_net = IncrementEmbeddingNet(
            input_dim=xs.shape[-1],
            trial_dim=neural_net_params.trial_dim,
            output_dim=neural_net_params.output_dim,
        )
        score_estimator = sbi.neural_nets.posterior_score_nn(
            sde_type="vp",
            embedding_net=embedding_net,
            z_score_x="none",
        )
    else:
        raise ValueError(f"Unknown neural net {neural_net_params.name}")

    inf = method(prior=prior_torch, score_estimator=score_estimator)

    # Perform training
    inf = inf.append_simulations(thetas, xs, exclude_invalid_x=False)
    train_kwargs = dict(cfg.method.params_train)
    _ = inf.train(**train_kwargs)

    # Build posterior
    posterior_kwargs = dict(cfg.method.params_build_posterior)
    posterior = inf.build_posterior(**posterior_kwargs)

    return NSEModel(posterior, cfg=cfg)


def convert_prior_to_torch(prior):
    if isinstance(prior, Normal):
        mu = torch.tensor(np.array(prior.mean))
        sigma = torch.tensor(np.array(prior.std))
        return dist.Independent(dist.Normal(mu, sigma), 1)
    else:
        raise ValueError(f"Unknown prior {prior}")


def h(a, b):
    # Distance correlation loss!
    N = a.shape[0]

    term1 = torch.linalg.norm(a - b, axis=-1)
    term2 = torch.linalg.norm(a[:, None, ...] - b, axis=-1).sum(axis=0) / (N - 2)
    term3 = torch.linalg.norm(a - b[:, None, ...], axis=-1).mean(axis=0) / (N - 2)
    term4 = torch.linalg.norm(a[None, ...] - b[:, None, ...], axis=-1).sum(axis=0) / (
        (N - 1) * (N - 2)
    )
    return term1 - term2 - term3 + term4


def summary_loss_fn(summary_net, thetas, xs):
    summary_stats = summary_net(xs)

    theta = thetas[::2]
    s = summary_stats[::2]
    theta_prime = thetas[1::2]
    s_prime = summary_stats[1::2]

    h_theta = h(theta, theta_prime)
    h_s = h(s, s_prime)

    term1 = torch.sum(h_theta * h_s)
    term2 = torch.sqrt(torch.sum(h_theta**2))
    term3 = torch.sqrt(torch.sum(h_s**2))

    return -term1 / (term2 * term3)


def run_npe_sufficient_summary_stat(cfg, task, data, rng_method):
    method = SNPE

    # Data
    thetas = data["thetas"]
    xs = data["xs"]

    # Data augmentation for different sequence lengths
    T_max = xs.shape[1]
    for t in range(2, T_max):
        idx = torch.randint(
            0,
            xs.shape[0],
            (
                int(
                    cfg.method.subseq_data_augmentation_fraction
                    * cfg.task.num_simulations
                    / (T_max - 1)
                ),
            ),
        )
        xs_subset = xs[idx]
        theta_subset = thetas[idx]
        xs_subset[:, t:] = torch.nan  # Cut off data after t
        xs = torch.cat([xs, xs_subset], dim=0)
        thetas = torch.cat([thetas, theta_subset], dim=0)

    idx = torch.randperm(xs.shape[0])

    thetas = thetas[idx]
    xs = xs[idx]

    # Setup inference
    prior = task.get_prior()
    prior_torch = convert_prior_to_torch(prior)

    neural_net_params = cfg.method.neural_net

    if neural_net_params.name == "rnn":
        embedding_net = TimeSeriesEmbeddingNet(
            input_dim=xs.shape[-1],
            output_dim=neural_net_params.output_dim,
            num_layers=neural_net_params.num_layers,
        )
        density_estimator = sbi.utils.posterior_nn(
            model=cfg.method.params_init.density_estimator,
        )
    elif neural_net_params.name == "increment":
        embedding_net = IncrementEmbeddingNet(
            input_dim=xs.shape[-1],
            trial_dim=neural_net_params.trial_dim,
            output_dim=neural_net_params.output_dim,
        )
        density_estimator = sbi.utils.posterior_nn(
            model=cfg.method.params_init.density_estimator,
        )
    else:
        raise ValueError(f"Unknown neural net {neural_net_params.name}")

    # First train the embedding network!
    print(thetas.shape)
    optimizer = torch.optim.Adam(embedding_net.parameters(), lr=1e-3)
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(thetas[:-1000], xs[:-1000]),
        batch_size=400,
        shuffle=True,
    )

    dataloader_val = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(thetas[-1000:], xs[-1000:]),
        batch_size=400,
        shuffle=False,
    )
    print(len(dataloader), len(dataloader_val))

    for epoch in range(100):
        for thetas_batch, xs_batch in dataloader:
            optimizer.zero_grad()
            loss = summary_loss_fn(embedding_net, thetas_batch, xs_batch)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            loss_val = 0
            for thetas_batch, xs_batch in dataloader_val:
                loss_val += summary_loss_fn(embedding_net, thetas_batch, xs_batch)
            loss_val /= len(dataloader_val)

        print(f"Epoch {epoch}, loss: {loss.item()}, loss_val: {loss_val}")

    embedding_net.eval()

    inf = method(prior=prior_torch, density_estimator=density_estimator)

    # Perform training
    summary_stats = embedding_net(xs).clone().detach()
    inf = inf.append_simulations(thetas, summary_stats, exclude_invalid_x=False)
    train_kwargs = dict(cfg.method.params_train)
    _ = inf.train(**train_kwargs)

    # Build posterior
    posterior_kwargs = dict(cfg.method.params_build_posterior)
    posterior = inf.build_posterior(**posterior_kwargs)

    return NPESummaryModel(posterior, embedding_net, cfg=cfg)


def run_nle_sufficient_summary_stat(cfg, task, data, rng_method):
    method = SNLE

    # Data
    thetas = data["thetas"]
    xs = data["xs"]

    # Data augmentation for different sequence lengths
    T_max = xs.shape[1]
    for t in range(2, T_max):
        idx = torch.randint(
            0,
            xs.shape[0],
            (
                int(
                    cfg.method.subseq_data_augmentation_fraction
                    * cfg.task.num_simulations
                    / (T_max - 1)
                ),
            ),
        )
        xs_subset = xs[idx]
        theta_subset = thetas[idx]
        xs_subset[:, t:] = torch.nan  # Cut off data after t
        xs = torch.cat([xs, xs_subset], dim=0)
        thetas = torch.cat([thetas, theta_subset], dim=0)

    idx = torch.randperm(xs.shape[0])

    thetas = thetas[idx]
    xs = xs[idx]

    # Setup inference
    prior = task.get_prior()
    prior_torch = convert_prior_to_torch(prior)

    neural_net_params = cfg.method.neural_net

    if neural_net_params.name == "rnn":
        embedding_net = TimeSeriesEmbeddingNet(
            input_dim=xs.shape[-1],
            output_dim=neural_net_params.output_dim,
            num_layers=neural_net_params.num_layers,
        )
        embedding_net = torch.nn.Sequential(
            embedding_net,
            torch.nn.Linear(neural_net_params.output_dim, neural_net_params.output_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(neural_net_params.output_dim, max(thetas.shape[-1] * 2, 5)),
        )
        density_estimator = sbi.utils.likelihood_nn(
            model=cfg.method.params_init.density_estimator,
        )
    elif neural_net_params.name == "increment":
        embedding_net = IncrementEmbeddingNet(
            input_dim=xs.shape[-1],
            trial_dim=neural_net_params.trial_dim,
            output_dim=neural_net_params.output_dim,
        )
        embedding_net = torch.nn.Sequential(
            embedding_net,
            torch.nn.Linear(neural_net_params.output_dim, neural_net_params.output_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(neural_net_params.output_dim, max(thetas.shape[-1] * 2, 5)),
        )
        density_estimator = sbi.utils.likelihood_nn(
            model=cfg.method.params_init.density_estimator,
        )
    else:
        raise ValueError(f"Unknown neural net {neural_net_params.name}")

    # First train the embedding network!
    print(thetas.shape)
    optimizer = torch.optim.Adam(embedding_net.parameters(), lr=1e-3)
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(thetas[:-1000], xs[:-1000]),
        batch_size=400,
        shuffle=True,
    )

    dataloader_val = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(thetas[-1000:], xs[-1000:]),
        batch_size=400,
        shuffle=False,
    )
    print(len(dataloader), len(dataloader_val))

    for epoch in range(100):
        for thetas_batch, xs_batch in dataloader:
            optimizer.zero_grad()
            loss = summary_loss_fn(embedding_net, thetas_batch, xs_batch)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            loss_val = 0
            for thetas_batch, xs_batch in dataloader_val:
                loss_val += summary_loss_fn(embedding_net, thetas_batch, xs_batch)
            loss_val /= len(dataloader_val)

        print(f"Epoch {epoch}, loss: {loss.item()}, loss_val: {loss_val}")

    embedding_net.eval()

    inf = method(prior=prior_torch, density_estimator=density_estimator)

    # Perform training
    summary_stats = embedding_net(xs).clone().detach()
    inf = inf.append_simulations(thetas, summary_stats, exclude_invalid_x=False)
    train_kwargs = dict(cfg.method.params_train)
    _ = inf.train(**train_kwargs)

    # Build posterior
    posterior_kwargs = dict(cfg.method.params_build_posterior)
    posterior = inf.build_posterior(**posterior_kwargs)

    return NPESummaryModel(posterior, embedding_net, cfg=cfg)


def sliced_dc_loss_fn(summary_net, slice_net, thetas, xs):
    phi = torch.randn(thetas.shape)
    phi = phi / torch.linalg.norm(phi, axis=-1, keepdims=True)

    theta_sliced = torch.sum(phi * thetas, axis=-1, keepdims=True)
    summary_stats = summary_net(xs)
    summary_stats_extended = torch.concatenate(
        [summary_stats, torch.repeat_interleave(phi, 10, axis=-1)], axis=-1
    )
    summary_stats_sliced = slice_net(summary_stats_extended)

    theta = theta_sliced[::2]
    s = summary_stats_sliced[::2]
    theta_prime = theta_sliced[1::2]
    s_prime = summary_stats_sliced[1::2]

    h_theta = h(theta, theta_prime)
    h_s = h(s, s_prime)

    term1 = torch.sum(h_theta * h_s)
    term2 = torch.sqrt(torch.sum(h_theta**2))
    term3 = torch.sqrt(torch.sum(h_s**2))

    return -term1 / (term2 * term3)


def run_npe_sliced_sufficient_summary_stat(cfg, task, data, rng_method):
    method = SNPE

    # Data
    thetas = data["thetas"]
    xs = data["xs"]

    # Data augmentation for different sequence lengths
    T_max = xs.shape[1]
    for t in range(2, T_max):
        idx = torch.randint(
            0,
            xs.shape[0],
            (
                int(
                    cfg.method.subseq_data_augmentation_fraction
                    * cfg.task.num_simulations
                    / (T_max - 1)
                ),
            ),
        )
        xs_subset = xs[idx]
        theta_subset = thetas[idx]
        xs_subset[:, t:] = torch.nan  # Cut off data after t
        xs = torch.cat([xs, xs_subset], dim=0)
        thetas = torch.cat([thetas, theta_subset], dim=0)

    idx = torch.randperm(xs.shape[0])

    thetas = thetas[idx]
    xs = xs[idx]

    # Setup inference
    prior = task.get_prior()
    prior_torch = convert_prior_to_torch(prior)

    neural_net_params = cfg.method.neural_net

    if neural_net_params.name == "rnn":
        embedding_net = TimeSeriesEmbeddingNet(
            input_dim=xs.shape[-1],
            output_dim=neural_net_params.output_dim,
            num_layers=neural_net_params.num_layers,
        )
        density_estimator = sbi.utils.posterior_nn(
            model=cfg.method.params_init.density_estimator,
        )
    elif neural_net_params.name == "increment":
        embedding_net = IncrementEmbeddingNet(
            input_dim=xs.shape[-1],
            trial_dim=neural_net_params.trial_dim,
            output_dim=neural_net_params.output_dim,
        )
        density_estimator = sbi.utils.posterior_nn(
            model=cfg.method.params_init.density_estimator,
        )
    else:
        raise ValueError(f"Unknown neural net {neural_net_params.name}")

    input_dim = neural_net_params.output_dim + thetas.shape[-1] * 10
    sliced_summary_net = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 2),
    )

    # First train the embedding network!
    optimizer = torch.optim.Adam(embedding_net.parameters(), lr=1e-3)
    optimizer_slice = torch.optim.Adam(sliced_summary_net.parameters(), lr=1e-3)
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(thetas[:-1000], xs[:-1000]),
        batch_size=400,
        shuffle=True,
    )

    dataloader_val = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(thetas[-1000:], xs[-1000:]),
        batch_size=400,
        shuffle=False,
    )
    print(len(dataloader), len(dataloader_val))

    for epoch in range(100):
        for thetas_batch, xs_batch in dataloader:
            optimizer.zero_grad()
            optimizer_slice.zero_grad()
            loss = sliced_dc_loss_fn(
                embedding_net, sliced_summary_net, thetas_batch, xs_batch
            )
            loss.backward()
            optimizer.step()
            optimizer_slice.step()

        with torch.no_grad():
            loss_val = 0
            for thetas_batch, xs_batch in dataloader_val:
                loss_val += sliced_dc_loss_fn(
                    embedding_net, sliced_summary_net, thetas_batch, xs_batch
                )
            loss_val /= len(dataloader_val)

        print(f"Epoch {epoch}, loss: {loss.item()}, loss_val: {loss_val}")

    embedding_net.eval()
    sliced_summary_net.eval()

    inf = method(prior=prior_torch, density_estimator=density_estimator)

    # Perform training
    summary_stats = embedding_net(xs).clone().detach()

    inf = inf.append_simulations(thetas, summary_stats, exclude_invalid_x=False)
    train_kwargs = dict(cfg.method.params_train)
    _ = inf.train(**train_kwargs)

    # Build posterior
    posterior_kwargs = dict(cfg.method.params_build_posterior)
    posterior = inf.build_posterior(**posterior_kwargs)

    return NPESummaryModel(posterior, embedding_net, cfg=cfg)


def run_nle_sliced_sufficient_summary_stat(cfg, task, data, rng_method):
    method = SNLE

    # Data
    thetas = data["thetas"]
    xs = data["xs"]

    # Data augmentation for different sequence lengths
    T_max = xs.shape[1]
    for t in range(2, T_max):
        idx = torch.randint(
            0,
            xs.shape[0],
            (
                int(
                    cfg.method.subseq_data_augmentation_fraction
                    * cfg.task.num_simulations
                    / (T_max - 1)
                ),
            ),
        )
        xs_subset = xs[idx]
        theta_subset = thetas[idx]
        xs_subset[:, t:] = torch.nan  # Cut off data after t
        xs = torch.cat([xs, xs_subset], dim=0)
        thetas = torch.cat([thetas, theta_subset], dim=0)

    idx = torch.randperm(xs.shape[0])

    thetas = thetas[idx]
    xs = xs[idx]

    # Setup inference
    prior = task.get_prior()
    prior_torch = convert_prior_to_torch(prior)

    neural_net_params = cfg.method.neural_net

    if neural_net_params.name == "rnn":
        embedding_net = TimeSeriesEmbeddingNet(
            input_dim=xs.shape[-1],
            output_dim=neural_net_params.output_dim,
            num_layers=neural_net_params.num_layers,
        )
        embedding_net = torch.nn.Sequential(
            embedding_net,
            torch.nn.Linear(neural_net_params.output_dim, neural_net_params.output_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(neural_net_params.output_dim, max(thetas.shape[-1] * 2, 5)),
        )
        density_estimator = sbi.utils.likelihood_nn(
            model=cfg.method.params_init.density_estimator,
        )
    elif neural_net_params.name == "increment":
        embedding_net = IncrementEmbeddingNet(
            input_dim=xs.shape[-1],
            trial_dim=neural_net_params.trial_dim,
            output_dim=neural_net_params.output_dim,
        )
        embedding_net = torch.nn.Sequential(
            embedding_net,
            torch.nn.Linear(neural_net_params.output_dim, neural_net_params.output_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(neural_net_params.output_dim, max(thetas.shape[-1] * 2, 5)),
        )
        density_estimator = sbi.utils.likelihood_nn(
            model=cfg.method.params_init.density_estimator,
        )
    else:
        raise ValueError(f"Unknown neural net {neural_net_params.name}")

    input_dim = max(thetas.shape[-1] * 2, 5) + thetas.shape[-1] * 10
    sliced_summary_net = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 2),
    )

    # First train the embedding network!
    optimizer = torch.optim.Adam(embedding_net.parameters(), lr=1e-3)
    optimizer_slice = torch.optim.Adam(sliced_summary_net.parameters(), lr=1e-3)
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(thetas[:-1000], xs[:-1000]),
        batch_size=400,
        shuffle=True,
    )

    dataloader_val = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(thetas[-1000:], xs[-1000:]),
        batch_size=400,
        shuffle=False,
    )
    print(len(dataloader), len(dataloader_val))

    for epoch in range(100):
        for thetas_batch, xs_batch in dataloader:
            optimizer.zero_grad()
            optimizer_slice.zero_grad()
            loss = sliced_dc_loss_fn(
                embedding_net, sliced_summary_net, thetas_batch, xs_batch
            )
            loss.backward()
            optimizer.step()
            optimizer_slice.step()

        with torch.no_grad():
            loss_val = 0
            for thetas_batch, xs_batch in dataloader_val:
                loss_val += sliced_dc_loss_fn(
                    embedding_net, sliced_summary_net, thetas_batch, xs_batch
                )
            loss_val /= len(dataloader_val)

        print(f"Epoch {epoch}, loss: {loss.item()}, loss_val: {loss_val}")

    embedding_net.eval()
    sliced_summary_net.eval()

    inf = method(prior=prior_torch, density_estimator=density_estimator)

    # Perform training
    summary_stats = embedding_net(xs).clone().detach()

    inf = inf.append_simulations(thetas, summary_stats, exclude_invalid_x=False)
    train_kwargs = dict(cfg.method.params_train)
    _ = inf.train(**train_kwargs)

    # Build posterior
    posterior_kwargs = dict(cfg.method.params_build_posterior)
    posterior = inf.build_posterior(**posterior_kwargs)

    return NPESummaryModel(posterior, embedding_net, cfg=cfg)
