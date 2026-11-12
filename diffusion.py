__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

"""
This code is based on @dome272 implementation of DDPM's
https://github.com/dome272/Diffusion-Models-pytorch
"""

from utils import *


class Diffusion:
    def __init__(self, noise_steps, img_size, beta_start, beta_end, device):
        self.noise_steps = noise_steps
        
        self.beta = self.linear_noise_schedule(beta_start, beta_end).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def linear_noise_schedule(self, beta_start, beta_end):
        return torch.linspace(beta_start, beta_end, self.noise_steps)

    def noise_images(self, x, t, pyramid=False, simplex=False, discount = 0.8):
        """This method generates the latent representations at time t of the input images. Equation (1) in ANDi paper

        Parameters
        ----------
        x : tensor
            Input x_0
        t : tensor
            tensor containing the values for the time steps t
        pyramid : bool, optional
            flag that decides if pyramid noise is used, by default False
        simplex : bool, optional
            flag that decides if simplex noise is used, by default False
        discount : float, optional
            the discount for the pyramid noise, by default 0.8

        Returns
        -------
        tensor, tensor
            The first tensor contains the latent representations, the second the noise used (needed for training)
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[
            :, None, None, None
        ]
        if simplex == True:
            raise RuntimeError("simplex noise was removed; set simplex: False")
        elif pyramid == True:
            noise = pyramid_noise_like(x.shape[0], x.shape[1], self.img_size, discount, x.device)
        else:
            noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def ddpm_mu_t(self, x, predicted_noise, t):
        """This method calculates the mean of the Gaussian transition at time t given the noise. ANDi paper notation mu_q, or mu_theta.

        Parameters
        ----------
        x : tensor
            The tensor containing x_t
        predicted_noise : tensor
            The tensor containing the noise (either predicted or ground truth)
        t : tensor
            The tensor containing the time steps

        Returns
        -------
        tensor
            The tensor contatining x_t-1
        """
        alpha = self.alpha[t][:, None, None, None]
        alpha_hat = self.alpha_hat[t][:, None, None, None]
        return (
            1
            / torch.sqrt(alpha)
            * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise)
        )

    def ddpm_mean_t(self, x, t, predicted_noise=None, x_0=None):
        """This method calculates the mean of the Gaussian transition at time t given the predicted noise, or the original sample x_0.
        Additionally clips the predictions like the original implementation of DDPM's, when the noise is provided to calculate the mean.

        Parameters
        ----------
        x : tensor
            The tensor containing x_t
        predicted_noise : tensor
            The tensor containing the noise (either predicted or ground truth)
        t : tensor
            The tensor containing the time steps
        x_0 : tensor
            The tensor containing x_0

        Returns
        -------
        tensor
            The tensor contatining x_t-1
        """
        if predicted_noise == None and x_0 == None:
            print("Either noise or x_0 have to be given to calculate x_t-1.")
            exit(1)
        alpha = self.alpha[t][:, None, None, None]
        alpha_hat = self.alpha_hat[t][:, None, None, None]
        beta = self.beta[t][:, None, None, None]
        alpha_hat_minus_one = self.alpha_hat[t - 1][:, None, None, None]
        if x_0 == None:
            pred_x0 = (
                1 / torch.sqrt(alpha_hat) * x
                - torch.sqrt((1 - alpha_hat) / (alpha_hat)) * predicted_noise
            )
            pred_x0 = pred_x0.clamp(-1, 1)
            x_0 = pred_x0
        w0 = torch.sqrt(alpha_hat_minus_one) * beta / (1 - alpha_hat)
        wt = torch.sqrt(alpha) * (1 - alpha_hat_minus_one) / (1 - alpha_hat)
        return w0 * x_0 + wt * x

    def sample(
        self, model, n, channels, pyramid=False, simplex=False, discount = 0.8
    ):
        """This method samples from the learned DDPM.

        Parameters
        ----------
        model : _type_
            The learned DDPM model (U-Net)
        n : int
            The number of samples you want to create
        channels : int
            The number of channels the model is trained on. For ANDi, this is 4 (FLAIR, T1, T1ce, T2)
        pyramid : bool, optional
            flag that decides if pyramid noise is used, by default False
        simplex : bool, optional
            flag that decides if simplex noise is used, by default False
        discount : float, optional
            the discount for the pyramid noise, by default 0.8

        Returns
        -------
        tensor
            The samples created.
        """
        model.eval()
        with torch.no_grad():
            if simplex == True:
                raise RuntimeError("simplex noise was removed; set simplex: False")
            elif pyramid == True:
                x = pyramid_noise_like(n, channels, self.img_size, discount, self.device)
            else:
                x = torch.randn((n, channels, self.img_size, self.img_size)).to(
                    self.device
                )
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                alpha_hat_minus_one = self.alpha_hat[t - 1][:, None, None, None]
                if i > 1:
                    if simplex == True:
                        raise RuntimeError("simplex noise was removed; set simplex: False")
                    elif pyramid == True:
                        noise = pyramid_noise_like(n, channels, self.img_size, discount, x.device)
                    else:
                        noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                var = beta * (1 - alpha_hat_minus_one) / (1 - alpha_hat)
                x = self.ddpm_mu_t(x, predicted_noise, t) + torch.sqrt(var) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def sample_controlnet(
        self,
        model,
        n,
        channels,
        control,
        cfg_scale=7.5,
        pyramid=True,
        simplex=False,
        discount=0.8,
    ):
        """DDPM sampling with ControlNet + classifier-free guidance (CFG).

        model must implement forward(x, t, control=..., cfg_scale=...) like ControlNetUNet.

        Parameters
        ----------
        control : tensor
            (n, control_channels, H, W) conditioning maps (e.g. selected modalities).
        cfg_scale : float
            Guidance strength: eps = eps_uncond + scale * (eps_cond - eps_uncond).
        pyramid, simplex, discount : same as sample().
        """
        model.eval()
        control = control.to(self.device)
        if control.shape[0] != n:
            raise ValueError(
                f"control batch {control.shape[0]} must equal n={n}"
            )
        with torch.no_grad():
            if simplex:
                raise RuntimeError("simplex noise was removed; set simplex: False")
            elif pyramid:
                x = pyramid_noise_like(n, channels, self.img_size, discount, self.device)
            else:
                x = torch.randn((n, channels, self.img_size, self.img_size)).to(
                    self.device
                )
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(
                    x, t, control=control, cfg_scale=cfg_scale
                )
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                alpha_hat_minus_one = self.alpha_hat[t - 1][:, None, None, None]
                if i > 1:
                    if simplex:
                        raise RuntimeError("simplex noise was removed; set simplex: False")
                    elif pyramid:
                        noise = pyramid_noise_like(
                            n, channels, self.img_size, discount, x.device
                        )
                    else:
                        noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                var = beta * (1 - alpha_hat_minus_one) / (1 - alpha_hat)
                x = self.ddpm_mu_t(x, predicted_noise, t) + torch.sqrt(var) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def sample_from_initial(
        self,
        model,
        x_t,
        t_start,
        pyramid=False,
        simplex=False,
        discount=0.8,
        return_float=False,
    ):
        """Sample from a given initial x_t down to x_0 (denoised image).

        You provide the noisy state x_t at timestep t_start; the reverse process
        runs from t_start down to 1 and returns the corresponding x_0.

        Parameters
        ----------
        model : nn.Module
            The learned DDPM model (U-Net).
        x_t : tensor
            Initial noisy state at timestep t_start. Shape (n, channels, H, W).
            Will be moved to self.device if needed.
        t_start : int
            Timestep of the given x_t (1 <= t_start <= noise_steps).
        pyramid : bool, optional
            Use pyramid noise for sampling steps, by default False.
        simplex : bool, optional
            Use simplex noise for sampling steps, by default False.
        discount : float, optional
            Discount for pyramid noise, by default 0.8.
        return_float : bool, optional
            If True, return x_0 in [-1, 1] float; if False, return uint8 in [0, 255], by default False.

        Returns
        -------
        tensor
            Sampled x_0. Either float in [-1, 1] or uint8 in [0, 255] depending on return_float.
        """
        if not (1 <= t_start <= self.noise_steps):
            raise ValueError(
                f"t_start must be in [1, {self.noise_steps}], got {t_start}."
            )
        x_t = x_t.to(self.device)
        n = x_t.shape[0]
        model.eval()
        with torch.no_grad():
            x = x_t
            for i in tqdm(reversed(range(1, t_start + 1)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                alpha_hat_minus_one = self.alpha_hat[t - 1][:, None, None, None]
                if i > 1:
                    if simplex:
                        raise RuntimeError("simplex noise was removed; set simplex: False")
                    elif pyramid:
                        noise = pyramid_noise_like(
                            n, x.shape[1], self.img_size, discount, x.device
                        )
                    else:
                        noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                var = beta * (1 - alpha_hat_minus_one) / (1 - alpha_hat)
                x = self.ddpm_mu_t(x, predicted_noise, t) + torch.sqrt(var) * noise
        model.train()
        x = x.clamp(-1, 1)
        if return_float:
            return x
        x = (x + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def sample_controlnet_from_initial(
        self,
        model,
        x_t,
        t_start,
        control,
        cfg_scale=7.5,
        pyramid=False,
        simplex=False,
        discount=0.8,
        return_float=False,
    ):
        """Denoise from x_t down to x_0 with a ControlNetUNet (same schedule as ``sample_from_initial``).

        Parameters
        ----------
        control : tensor
            (n, C_ctl, H, W) aligned with ``x_t`` batch.
        cfg_scale : float
            Passed to ``model(..., cfg_scale=...)`` each step (CFG at inference).
        """
        if not (1 <= t_start <= self.noise_steps):
            raise ValueError(
                f"t_start must be in [1, {self.noise_steps}], got {t_start}."
            )
        x_t = x_t.to(self.device)
        control = control.to(self.device)
        n = x_t.shape[0]
        if control.shape[0] != n:
            raise ValueError(
                f"control batch {control.shape[0]} != x_t batch {n}"
            )
        model.eval()
        with torch.no_grad():
            x = x_t
            for i in tqdm(reversed(range(1, t_start + 1)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(
                    x, t, control=control, cfg_scale=cfg_scale
                )
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                alpha_hat_minus_one = self.alpha_hat[t - 1][:, None, None, None]
                if i > 1:
                    if simplex:
                        raise RuntimeError("simplex noise was removed; set simplex: False")
                    elif pyramid:
                        noise = pyramid_noise_like(
                            n, x.shape[1], self.img_size, discount, x.device
                        )
                    else:
                        noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                var = beta * (1 - alpha_hat_minus_one) / (1 - alpha_hat)
                x = self.ddpm_mu_t(x, predicted_noise, t) + torch.sqrt(var) * noise
        model.train()
        x = x.clamp(-1, 1)
        if return_float:
            return x
        x = (x + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def sample_controlnet_from_initial_dual(
        self,
        model,
        x_t,
        t_start,
        control,
        cfg_scale_small,
        cfg_scale_large,
        pyramid=False,
        simplex=False,
        discount=0.8,
        return_float=False,
    ):
        """Same reverse chain as ``sample_controlnet_from_initial``, but two CFG scales.

        Both chains start from the same ``x_t`` and share the stochastic noise at
        every step so the reconstructions are comparable. Requires
        ``cfg_scale_small < 1 < cfg_scale_large``.

        Returns
        -------
        tuple
            ``(x0_small, x0_large)``, each float in [-1, 1] or uint8 in [0, 255].
        """
        cfg_scale_small = float(cfg_scale_small)
        cfg_scale_large = float(cfg_scale_large)
        if not (cfg_scale_small < 1.0 < cfg_scale_large):
            raise ValueError(
                "dual CFG requires cfg_scale_small < 1 < cfg_scale_large, "
                f"got small={cfg_scale_small}, large={cfg_scale_large}"
            )
        if not (1 <= t_start <= self.noise_steps):
            raise ValueError(
                f"t_start must be in [1, {self.noise_steps}], got {t_start}."
            )
        x_t = x_t.to(self.device)
        control = control.to(self.device)
        n = x_t.shape[0]
        if control.shape[0] != n:
            raise ValueError(
                f"control batch {control.shape[0]} != x_t batch {n}"
            )
        model.eval()
        with torch.no_grad():
            x_s = x_t.clone()
            x_l = x_t.clone()
            for i in tqdm(reversed(range(1, t_start + 1)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_s = model(
                    x_s, t, control=control, cfg_scale=cfg_scale_small
                )
                predicted_l = model(
                    x_l, t, control=control, cfg_scale=cfg_scale_large
                )
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                alpha_hat_minus_one = self.alpha_hat[t - 1][:, None, None, None]
                if i > 1:
                    if simplex:
                        raise RuntimeError("simplex noise was removed; set simplex: False")
                    elif pyramid:
                        noise = pyramid_noise_like(
                            n, x_t.shape[1], self.img_size, discount, x_t.device
                        )
                    else:
                        noise = torch.randn_like(x_t)
                else:
                    noise = torch.zeros_like(x_t)
                var = beta * (1 - alpha_hat_minus_one) / (1 - alpha_hat)
                std = torch.sqrt(var)
                x_s = self.ddpm_mu_t(x_s, predicted_s, t) + std * noise
                x_l = self.ddpm_mu_t(x_l, predicted_l, t) + std * noise
        model.train()
        x_s = x_s.clamp(-1, 1)
        x_l = x_l.clamp(-1, 1)
        if return_float:
            return x_s, x_l
        x_s = ((x_s + 1) / 2 * 255).type(torch.uint8)
        x_l = ((x_l + 1) / 2 * 255).type(torch.uint8)
        return x_s, x_l
    
    def sample_controlnet_from_initial_cond_uncond(
        self,
        model,
        x_t,
        t_start,
        control,
        pyramid=False,
        simplex=False,
        discount=0.8,
        return_float=False,
    ):
        """Same reverse chain as ``sample_controlnet_from_initial``, cond vs uncond.

        Both chains start from the same ``x_t`` and share the stochastic noise at
        every step. Cond uses the given control map; uncond uses zeros.

        Returns
        -------
        tuple
            ``(x0_cond, x0_uncond)``, each float in [-1, 1] or uint8 in [0, 255].
        """
        if not (1 <= t_start <= self.noise_steps):
            raise ValueError(
                f"t_start must be in [1, {self.noise_steps}], got {t_start}."
            )
        x_t = x_t.to(self.device)
        control = control.to(self.device)
        n = x_t.shape[0]
        if control.shape[0] != n:
            raise ValueError(
                f"control batch {control.shape[0]} != x_t batch {n}"
            )
        zeros = torch.zeros_like(control)
        model.eval()
        with torch.no_grad():
            x_c = x_t.clone()
            x_u = x_t.clone()
            for i in tqdm(reversed(range(1, t_start + 1)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_c = model(
                    x_c, t, control=control, cfg_scale=1.0
                )
                predicted_u = model(
                    x_u, t, control=zeros, cfg_scale=1.0
                )
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                alpha_hat_minus_one = self.alpha_hat[t - 1][:, None, None, None]
                if i > 1:
                    if simplex:
                        raise RuntimeError("simplex noise was removed; set simplex: False")
                    elif pyramid:
                        noise = pyramid_noise_like(
                            n, x_t.shape[1], self.img_size, discount, x_t.device
                        )
                    else:
                        noise = torch.randn_like(x_t)
                else:
                    noise = torch.zeros_like(x_t)
                var = beta * (1 - alpha_hat_minus_one) / (1 - alpha_hat)
                std = torch.sqrt(var)
                x_c = self.ddpm_mu_t(x_c, predicted_c, t) + std * noise
                x_u = self.ddpm_mu_t(x_u, predicted_u, t) + std * noise
        model.train()
        x_c = x_c.clamp(-1, 1)
        x_u = x_u.clamp(-1, 1)
        if return_float:
            return x_c, x_u
        x_c = ((x_c + 1) / 2 * 255).type(torch.uint8)
        x_u = ((x_u + 1) / 2 * 255).type(torch.uint8)
        return x_c, x_u
    
    def normative_diffusion(
        self,
        model,
        images,
        start=75,
        stop=200,
        pyramid=False,
        discount=0.8,
        sign_info=False,
    ):
        """This method calculates the deviations for each time step t in the interval T_l, T_u.
        In the ANDi paper this method corresponds to line 3 to 9 in the pseudo-code.

        Parameters
        ----------
        model : _type_
            The learned DDPM model (U-Net)
        images : tensor
            The samples x_0
        start : int, optional
            Lower endpoint of interval T_l, by default 75
        stop : int, optional
            Upper endpoint of interval T_u, by default 200
        pyramid : bool, optional
            flag that decides if pyramid noise is used, by default False
        discount : float, optional
            the discount for the pyramid noise, by default 0.8
        sign_info : bool, optional
            If False (default), use squared deviation `(mu_q - mu_theta)**2`.
            If True, use a one-sided signed signal and ReLU:
            ``reconstruct=True`` → ``relu(mu_theta - mu_q)``.
        Returns
        -------
        tensor
            The tensor that contains the deviations for each time step in the dimension with the index 1
        """
        if stop == None:
            stop = self.noise_steps
        if start == 0:  # The start can not be the original sample x_0
            start = 1
        num_images = images.shape[0]
        model.eval()
        with torch.no_grad():
            dts = torch.zeros(
                (
                    num_images,
                    stop - start,
                    images.shape[1],
                    images.shape[2],
                    images.shape[3],
                )
            ).to(self.device)

            # calculate the deviations
            for i in tqdm(reversed(range(start, stop)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                x_t, noise = self.noise_images(images, t, pyramid=pyramid, discount=discount)
                predicted_noise = model(x_t, t)
                mu_theta = self.ddpm_mu_t(x_t, predicted_noise, t)
                mu_q = self.ddpm_mu_t(x_t, noise, t)
                if not sign_info:
                    d_t = (mu_q - mu_theta) ** 2
                else: # stk: in this case, it can only be reconstruction mode
                    d_t = torch.relu(mu_theta - mu_q) # stk: specifically for HIE lesion
                dts[:, i - start] = d_t
        return dts

    def normative_diffusion_controlnet(
        self,
        model,
        images,
        control,
        start=75,
        stop=200,
        pyramid=False,
        discount=0.8,
        cfg_scale=None,
        sign_info=False,
        use_square=False,
    ):
        """Same as ``normative_diffusion``, but for ``ControlNetUNet`` that requires ``control``.

        Parameters
        ----------
        model : nn.Module
            Must implement ``forward(x, t, control=..., cfg_scale=...)``.
        images : tensor
            Clean samples x_0 (B, C, H, W) in [-1, 1].
        control : tensor
            Conditioning (B, C_ctl, H, W), same spatial size as ``images``.
        cfg_scale : float, optional
            If set and ``model`` is in eval mode, uses CFG (cond vs uncond). If ``None``,
            runs a single forward pass with the given ``control``.
        sign_info: same as ``normative_diffusion``. 
            ``reconstruct=False`` → ``relu(mu_q - mu_theta)``;
            ``reconstruct=True`` → ``relu(mu_theta - mu_q)``.
        reconstruct : bool, optional
            Only used when ``sign_info`` is True (ANDi squared mode ignores this).
        """
        if stop is None:
            stop = self.noise_steps
        if start == 0:
            start = 1
        num_images = images.shape[0]
        control = control.to(device=self.device, dtype=images.dtype)
        model.eval()

        reconstruct = (not cfg_scale is None) and (cfg_scale < 1) # stk: if cfg_scale < 1, it is reconstruction mode. 
                                                               # if cfg_scale is None, it is equivalent to cfg_scale = 1
        
        with torch.no_grad():
            dts = torch.zeros(
                (
                    num_images,
                    stop - start,
                    images.shape[1],
                    images.shape[2],
                    images.shape[3],
                )
            ).to(self.device)

            for i in tqdm(reversed(range(start, stop)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                x_t, noise = self.noise_images(
                    images, t, pyramid=pyramid, discount=discount
                )
                predicted_noise = model(
                    x_t, t, control=control, cfg_scale=cfg_scale
                )
                mu_theta = self.ddpm_mu_t(x_t, predicted_noise, t)
                mu_q = self.ddpm_mu_t(x_t, noise, t)
                if not sign_info:
                    d_t = (mu_q - mu_theta) ** 2
                elif not reconstruct:
                    d_t = torch.relu(mu_q - mu_theta)
                    if use_square:
                        d_t = d_t ** 2
                else:
                    d_t = torch.relu(mu_theta - mu_q)
                    if use_square:
                        d_t = d_t ** 2
                dts[:, i - start] = d_t
        return dts
    
    def normative_diffusion_controlnet_chain(
        self,
        model,
        images,
        control,
        start=75,
        stop=200,
        pyramid=False,
        discount=0.8,
        cfg_scale=None,
        sign_info=False,
    ):
        """Sequential ControlNet reverse: one noise at t=stop-1, then denoise to start.

        Same per-step distance as ``normative_diffusion_controlnet`` (vs true posterior
        from x_0), but the predicted state is carried forward instead of re-noising
        independently at each t. Variance is zero (mean-only reverse).
        """
        if stop is None:
            stop = self.noise_steps
        if start == 0:
            start = 1
        num_images = images.shape[0]
        control = control.to(device=self.device, dtype=images.dtype)
        model.eval()
        reconstruct = (cfg_scale is not None) and (cfg_scale < 1)

        with torch.no_grad():
            dts = torch.zeros(
                (
                    num_images,
                    stop - start,
                    images.shape[1],
                    images.shape[2],
                    images.shape[3],
                )
            ).to(self.device)

            t0 = (torch.ones(num_images) * (stop - 1)).long().to(self.device)
            x_t, _noise = self.noise_images(
                images, t0, pyramid=pyramid, discount=discount
            )
            predicted_chain = x_t
            correct_chain = x_t

            for i in tqdm(reversed(range(start, stop)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                predicted_noise = model(
                    predicted_chain, t, control=control, cfg_scale=cfg_scale
                )
                mu_theta = self.ddpm_mu_t(predicted_chain, predicted_noise, t)
                mu_q = self.ddpm_mean_t(correct_chain, t, x_0=images)
                if not sign_info:
                    d_t = (mu_q - mu_theta) ** 2
                elif not reconstruct:
                    d_t = torch.relu(mu_q - mu_theta)
                else:
                    d_t = torch.relu(mu_theta - mu_q)
                dts[:, i - start] = d_t
                predicted_chain = mu_theta
                correct_chain = mu_q
        return dts
    
    def normative_diffusion_controlnet_dual_cfg_chain(
        self,
        model,
        images,
        control,
        start=75,
        stop=200,
        pyramid=False,
        discount=0.8,
        cfg_scale_small=0.5,
        cfg_scale_large=7.5,
        sign_info=False,
    ):
        """Sequential dual-CFG reverse: one noise at t=stop-1, two chains to start.

        Per step: d_t from (mu_small - mu_large); square if sign_info is False,
        relu if True. After the first step the two states diverge, so this is not
        a rescaling of single-CFG / (eps_cond - eps_uncond). Variance is zero.
        """
        cfg_scale_small = float(cfg_scale_small)
        cfg_scale_large = float(cfg_scale_large)
        if not (cfg_scale_small < 1.0 < cfg_scale_large):
            raise ValueError(
                "dual CFG requires cfg_scale_small < 1 < cfg_scale_large, "
                f"got small={cfg_scale_small}, large={cfg_scale_large}"
            )
        if stop is None:
            stop = self.noise_steps
        if start == 0:
            start = 1
        num_images = images.shape[0]
        control = control.to(device=self.device, dtype=images.dtype)
        model.eval()

        with torch.no_grad():
            dts = torch.zeros(
                (
                    num_images,
                    stop - start,
                    images.shape[1],
                    images.shape[2],
                    images.shape[3],
                )
            ).to(self.device)

            t0 = (torch.ones(num_images) * (stop - 1)).long().to(self.device)
            x_t, _noise = self.noise_images(
                images, t0, pyramid=pyramid, discount=discount
            )
            x_small = x_t.clone()
            x_large = x_t.clone()

            for i in tqdm(reversed(range(start, stop)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                eps_small = model(
                    x_small, t, control=control, cfg_scale=cfg_scale_small
                )
                eps_large = model(
                    x_large, t, control=control, cfg_scale=cfg_scale_large
                )
                mu_small = self.ddpm_mu_t(x_small, eps_small, t)
                mu_large = self.ddpm_mu_t(x_large, eps_large, t)
                diff = mu_small - mu_large
                if not sign_info:
                    d_t = diff ** 2
                else:
                    d_t = torch.relu(diff)
                dts[:, i - start] = d_t
                x_small = mu_small
                x_large = mu_large
        return dts
    
    def normative_diffusion_controlnet_eps(
        self,
        model,
        images,
        control,
        start=75,
        stop=200,
        pyramid=False,
        discount=0.8,
        sign_info=False,
        use_square=False,
        use_true_noise=False,
        use_clip=True
    ):
        """ANDi accumulation in noise space: ``eps_cond`` vs ``eps_uncond``.

        At each ``t`` the same independently noised ``x_t`` is scored with
        control present vs a zero control map. Unlike
        ``normative_diffusion_controlnet`` / ``*_dual_cfg`` (which compare
        ``ddpm_mu_t`` means), the per-step map is the predicted-noise residual:

        - ``sign_info=False``: ``(eps_cond - eps_uncond)**2``
        - ``sign_info=True``: ``relu(eps_cond - eps_uncond)``
        
        - use_true_noise=True: for unconditional case, use the gaussian noise from forward
        """
        if stop is None:
            stop = self.noise_steps
        if start == 0:
            start = 1
        num_images = images.shape[0]
        control = control.to(device=self.device, dtype=images.dtype)
        model.eval()
        inner = model.module if hasattr(model, "module") else model

        with torch.no_grad():
            dts = torch.zeros(
                (
                    num_images,
                    stop - start,
                    images.shape[1],
                    images.shape[2],
                    images.shape[3],
                )
            ).to(self.device)

            for i in tqdm(reversed(range(start, stop)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                x_t, _noise = self.noise_images(
                    images, t, pyramid=pyramid, discount=discount
                )
                if hasattr(inner, "predict_eps_cond_uncond"):
                    eps_cond, eps_uncond = inner.predict_eps_cond_uncond(
                        x_t, t, control
                    )
                else:
                    zeros = torch.zeros_like(control)
                    eps_cond = model(
                        x_t, t, control=control, cfg_scale=1.0
                    )
                    if not use_true_noise:
                        eps_uncond = model(
                        x_t, t, control=zeros, cfg_scale=1.0
                    )

                diff = eps_cond - eps_uncond if not use_true_noise else eps_cond - _noise
                    
                if not sign_info:
                     d_t = diff ** 2
                else:
                    d_t = torch.relu(diff) if use_clip else diff
                       
                    d_t = d_t ** 2 if use_square else d_t
                        
                dts[:, i - start] = d_t
        return dts

    def normative_diffusion_controlnet_eps_chain(
        self,
        model,
        images,
        control,
        start=75,
        stop=200,
        pyramid=False,
        discount=0.8,
        sign_info=False,
        use_clip=True
        
    ):
        """Sequential reverse: one noise at t=stop-1, two mean-only chains.

        One chain uses control (``eps_cond``), the other zeros (``eps_uncond``).
        Per-step map is the same noise residual as
        ``normative_diffusion_controlnet_eps``. After the first step the two
        states diverge, so this is not a rescaling of the independent-t method.
        Variance is zero (mean-only reverse via ``ddpm_mu_t``).
        """
        if stop is None:
            stop = self.noise_steps
        if start == 0:
            start = 1
        num_images = images.shape[0]
        control = control.to(device=self.device, dtype=images.dtype)
        zeros = torch.zeros_like(control)
        model.eval()

        with torch.no_grad():
            dts = torch.zeros(
                (
                    num_images,
                    stop - start,
                    images.shape[1],
                    images.shape[2],
                    images.shape[3],
                )
            ).to(self.device)

            t0 = (torch.ones(num_images) * (stop - 1)).long().to(self.device)
            x_t, _noise = self.noise_images(
                images, t0, pyramid=pyramid, discount=discount
            )
            x_cond = x_t.clone()
            x_uncond = x_t.clone()

            for i in tqdm(reversed(range(start, stop)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                eps_cond = model(
                    x_cond, t, control=control, cfg_scale=1.0
                )
                eps_uncond = model(
                    x_uncond, t, control=zeros, cfg_scale=1.0
                )
                diff = eps_cond - eps_uncond
                if not sign_info:
                    d_t = diff ** 2
                else:
                    d_t = torch.relu(diff) if use_clip else diff
                dts[:, i - start] = d_t
                x_cond = self.ddpm_mu_t(x_cond, eps_cond, t)
                x_uncond = self.ddpm_mu_t(x_uncond, eps_uncond, t)
        return dts

    def normative_blocks(
        self, model, images, start=75, stop=200, skip=25, pyramid=False, discount = 0.8
    ):
        """Experimental method. Not described in ANDi paper.
        It calculates deviations for blocks of the diffusion markov chain with setting the variance to zero.

        Parameters
        ----------
        model : _type_
            The learned DDPM model (U-Net)
        images : tensor
            The samples x_0
        start : int, optional
            Lower endpoint of interval T_l, by default 75
        stop : int, optional
            Upper endpoint of interval T_u, by default 200
        skip : int, optional
            The number of deviations to skip before calculating the deviation for a block, by default 25
        pyramid : bool, optional
            flag that decides if pyramid noise is used, by default False
        discount : float, optional
            the discount for the pyramid noise, by default 0.8

        Returns
        -------
        tensor
            The tensor that contains the deviations for each block in the dimension with the index 1
        """
        if stop is None:
            stop = self.noise_steps
        if start == 0:  # The start can not be the original sample x_0
            start = 1
        num_images = images.shape[0]
        model.eval()
        with torch.no_grad():
            dts = torch.zeros(
                (
                    num_images,
                    int(((stop - start) / skip)),
                    images.shape[1],
                    images.shape[2],
                    images.shape[3],
                )
            ).to(self.device)

            t = (torch.ones(num_images) * stop - 1).long().to(self.device)
            x_t, noise = self.noise_images(images, t, pyramid=pyramid, discount = discount)
            correct_chain = x_t
            predicted_chain = x_t

            for i in tqdm(reversed(range(start, stop)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                predicted_noise = model(predicted_chain, t)
                predicted_chain = self.ddpm_mu_t(predicted_chain, predicted_noise, t)
                correct_chain = self.ddpm_mean_t(correct_chain, t, x_0=images)
                if i % skip == 0 or i == 1:
                    d_t = (correct_chain - predicted_chain) ** 2
                    dts[:, int((i - start) / skip)] = d_t
                    t = (torch.ones(num_images) * i - 1).long().to(self.device)
                    x_t, noise = self.noise_images(images, t, pyramid=pyramid, discount=discount)
                    predicted_chain = x_t
                    correct_chain = x_t
        return dts

    def deviations_noise(self, model, images, start=75, stop=200, pyramid=False, discount = 0.8):
        """This method calculates the deviations for each time step t in the interval T_l, T_u on the noise level.
        This provides similar results.

        Parameters
        ----------
        model : _type_
            The learned DDPM model (U-Net)
        images : tensor
            The samples x_0
        start : int, optional
            Lower endpoint of interval T_l, by default 75
        stop : int, optional
            Upper endpoint of interval T_u, by default 200
        pyramid : bool, optional
            flag that decides if pyramid noise is used, by default False
        discount : float, optional
            the discount for the pyramid noise, by default 0.8

        Returns
        -------
        tensor
            The tensor that contains the deviations for each time step in the dimension with the index 1
        """
        if stop is None:
            stop = self.noise_steps
        if start == 0:  # The start can not be the original sample x_0
            start = 1
        num_images = images.shape[0]
        model.eval()
        with torch.no_grad():
            dts = torch.zeros(
                (
                    num_images,
                    stop - start,
                    images.shape[1],
                    images.shape[2],
                    images.shape[3],
                )
            ).to(self.device)

            for i in tqdm(reversed(range(start, stop)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                x_t, noise = self.noise_images(images, t, pyramid=pyramid, discount=discount)
                predicted_noise = model(x_t, t)
                d_t = (predicted_noise - noise) ** 2
                dts[:, i - start] = d_t
        return dts

    def ano_ddpm(self, model, images, num_steps, simplex=False, pyramid=False, discount = 0.8):
        """This method implements the lesion localization from the AnoDDPM paper.
        It is faster than the original implementation for the simplex noise by creating dependent noise along the batch dimension.

        Parameters
        ----------
        model : _type_
            The learned DDPM model (U-Net)
        images : tensor
            The samples x_0
        num_steps : int
            The number of steps, the diffuion markov chain is used. Equal to T_u.
        simplex : bool, optional
            flag that decides if simplex noise is used, by default False
        pyramid : bool, optional
            flag that decides if pyramid noise is used, by default False
        discount : float, optional
            the discount for the pyramid noise, by default 0.8

        Returns
        -------
        tensor
            The pseudo-healthy created.
        """
        model.eval()
        num_images = images.shape[0]
        with torch.no_grad():
            t = (torch.ones(num_images) * num_steps).long().to(self.device)
            x, noise = self.noise_images(images, t, simplex=simplex, pyramid=pyramid, discount=discount)
            if simplex == True:
                raise RuntimeError("simplex noise was removed; set simplex: False")
            for i in tqdm(reversed(range(1, num_steps)), position=0):
                t = (torch.ones(num_images) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                alpha_hat_minus_one = self.alpha_hat[t - 1][:, None, None, None]
                if i > 1:
                    if simplex == True:
                        raise RuntimeError("simplex noise was removed; set simplex: False")
                    elif pyramid == True:
                        noise = pyramid_noise_like(x.shape[0], x.shape[1], self.img_size, discount, x.device)
                    else:
                        noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                var = beta * (1 - alpha_hat_minus_one) / (1 - alpha_hat)
                x = self.ddpm_mean_t(x, t, predicted_noise) + torch.sqrt(var) * noise
        return x
