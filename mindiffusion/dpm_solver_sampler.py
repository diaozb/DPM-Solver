import torch


class DPMSolverSampler:
    """
    DPM-Solver sampler for VP discrete-time DDPM.

    We use lambda = log(alpha / sigma).
    DDIM is DPM-Solver-1.
    DPM-Solver-2 uses 2 model evaluations per step.
    DPM-Solver-3 uses 3 model evaluations per step.
    """

    def __init__(self, ddpm):
        self.ddpm = ddpm
        self.eps_model = ddpm.eps_model
        self.n_T = ddpm.n_T

        device = next(ddpm.parameters()).device

        alphabar = ddpm.alphabar_t.detach().to(device)
        alphabar = alphabar.clamp(1e-12, 1.0 - 1e-12)

        self.alphabar = alphabar
        self.lambda_table = 0.5 * (
            torch.log(alphabar) - torch.log1p(-alphabar)
        )

        # We sample from t = n_T to t = 1.
        # t = 0 has sigma = 0, so lambda = +infinity.
        self.lambda_start = self.lambda_table[self.n_T].item()
        self.lambda_end = self.lambda_table[1].item()

    def _to_tensor(self, x, device):
        if torch.is_tensor(x):
            return x.to(device)
        return torch.tensor(x, device=device, dtype=torch.float32)

    def alpha(self, lamb):
        # VP: alpha^2 + sigma^2 = 1
        return torch.sqrt(torch.sigmoid(2.0 * lamb))

    def sigma(self, lamb):
        return torch.sqrt(torch.sigmoid(-2.0 * lamb))

    def t_from_lambda(self, lamb):
        """
        Convert lambda back to continuous discrete-time index t in [1, n_T].
        lambda_table is decreasing as t increases.
        """
        table = self.lambda_table

        if torch.is_tensor(lamb):
            lamb_value = lamb.detach()
        else:
            lamb_value = torch.tensor(lamb, device=table.device)

        # Use -lambda because -lambda_table[1:] is increasing.
        neg_table = -table[1:]
        pos = torch.searchsorted(neg_table, -lamb_value).item() + 1

        if pos <= 1:
            return 1.0
        if pos >= self.n_T:
            return float(self.n_T)

        lo = pos - 1
        hi = pos

        l_lo = table[lo]
        l_hi = table[hi]

        # linear interpolation in lambda
        w = (lamb_value - l_lo) / (l_hi - l_lo)
        t = lo + w * (hi - lo)
        return float(t.item())

    def model_fn(self, x, lamb):
        """
        eps_theta(x_t, t).
        The minDiffusion U-Net accepts normalized time t / n_T.
        """
        t_idx = self.t_from_lambda(lamb)
        t_norm = torch.full(
            (x.shape[0],),
            t_idx / self.n_T,
            device=x.device,
            dtype=x.dtype,
        )
        return self.eps_model(x, t_norm)

    @torch.no_grad()
    def dpm_solver_1_step(self, x, lamb_s, lamb_t):
        """
        DPM-Solver-1, equivalent to DDIM.
        """
        device = x.device
        lamb_s = self._to_tensor(lamb_s, device)
        lamb_t = self._to_tensor(lamb_t, device)

        h = lamb_t - lamb_s

        alpha_s = self.alpha(lamb_s)
        alpha_t = self.alpha(lamb_t)
        sigma_t = self.sigma(lamb_t)

        eps_s = self.model_fn(x, lamb_s)

        x_t = (
            alpha_t / alpha_s * x
            - sigma_t * torch.expm1(h) * eps_s
        )
        return x_t

    @torch.no_grad()
    def dpm_solver_2_step(self, x, lamb_s, lamb_t):
        """
        DPM-Solver-2 midpoint version.
        """
        device = x.device
        lamb_s = self._to_tensor(lamb_s, device)
        lamb_t = self._to_tensor(lamb_t, device)

        h = lamb_t - lamb_s
        lamb_mid = lamb_s + 0.5 * h

        alpha_s = self.alpha(lamb_s)
        alpha_mid = self.alpha(lamb_mid)
        alpha_t = self.alpha(lamb_t)

        sigma_mid = self.sigma(lamb_mid)
        sigma_t = self.sigma(lamb_t)

        eps_s = self.model_fn(x, lamb_s)

        u = (
            alpha_mid / alpha_s * x
            - sigma_mid * torch.expm1(0.5 * h) * eps_s
        )

        eps_mid = self.model_fn(u, lamb_mid)

        x_t = (
            alpha_t / alpha_s * x
            - sigma_t * torch.expm1(h) * eps_mid
        )
        return x_t

    @torch.no_grad()
    def dpm_solver_3_step(self, x, lamb_s, lamb_t):
        """
        DPM-Solver-3.
        """
        device = x.device
        lamb_s = self._to_tensor(lamb_s, device)
        lamb_t = self._to_tensor(lamb_t, device)

        h = lamb_t - lamb_s
        r1 = 1.0 / 3.0
        r2 = 2.0 / 3.0

        lamb_1 = lamb_s + r1 * h
        lamb_2 = lamb_s + r2 * h

        alpha_s = self.alpha(lamb_s)
        alpha_1 = self.alpha(lamb_1)
        alpha_2 = self.alpha(lamb_2)
        alpha_t = self.alpha(lamb_t)

        sigma_1 = self.sigma(lamb_1)
        sigma_2 = self.sigma(lamb_2)
        sigma_t = self.sigma(lamb_t)

        eps_s = self.model_fn(x, lamb_s)

        u1 = (
            alpha_1 / alpha_s * x
            - sigma_1 * torch.expm1(r1 * h) * eps_s
        )

        eps_1 = self.model_fn(u1, lamb_1)
        D1 = eps_1 - eps_s

        coef_u2 = (r2 / r1) * (
            torch.expm1(r2 * h) / (r2 * h) - 1.0
        )

        u2 = (
            alpha_2 / alpha_s * x
            - sigma_2 * torch.expm1(r2 * h) * eps_s
            - sigma_2 * coef_u2 * D1
        )

        eps_2 = self.model_fn(u2, lamb_2)
        D2 = eps_2 - eps_s

        coef_x = (1.0 / r2) * (
            torch.expm1(h) / h - 1.0
        )

        x_t = (
            alpha_t / alpha_s * x
            - sigma_t * torch.expm1(h) * eps_s
            - sigma_t * coef_x * D2
        )

        return x_t

    def get_orders(self, method, nfe):
        """
        Return per-step solver orders.
        Sum of orders = NFE.
        """
        if method == "ddim":
            return [1] * nfe

        if method == "dpm_solver_2":
            orders = [2] * (nfe // 2)
            if nfe % 2 == 1:
                orders.append(1)
            return orders

        if method == "dpm_solver_3":
            orders = [3] * (nfe // 3)
            rem = nfe % 3
            if rem > 0:
                orders.append(rem)
            return orders

        raise ValueError(f"Unknown method: {method}")

    @torch.no_grad()
    def sample(self, n_sample, size, device, nfe, method):
        """
        method:
            'ddim'
            'dpm_solver_2'
            'dpm_solver_3'
        """
        self.ddpm.eval()

        x = torch.randn(n_sample, *size, device=device)

        orders = self.get_orders(method, nfe)

        lambdas = torch.linspace(
            self.lambda_start,
            self.lambda_end,
            len(orders) + 1,
            device=device,
        )

        for i, order in enumerate(orders):
            lamb_s = lambdas[i]
            lamb_t = lambdas[i + 1]

            if order == 1:
                x = self.dpm_solver_1_step(x, lamb_s, lamb_t)
            elif order == 2:
                x = self.dpm_solver_2_step(x, lamb_s, lamb_t)
            elif order == 3:
                x = self.dpm_solver_3_step(x, lamb_s, lamb_t)
            else:
                raise ValueError(order)

        return x.clamp(-1.0, 1.0)
