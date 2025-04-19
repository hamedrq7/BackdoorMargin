import numpy as np
import torch
import torch.nn as nn

from attacks.attack import Attack


class DeepFool(Attack):
    r"""
    'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks'
    [https://arxiv.org/abs/1511.04599]
    Distance Measure : L2
    Arguments:
        model (nn.Module): model to attack.
        steps (int): number of steps. (Default: 50)
        overshoot (float): parameter for enhancing the noise. (Default: 0.02)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        >>> attack = torchattacks.DeepFool(model, steps=50, overshoot=0.02)
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, model,
                 steps: int=100,
                 overshoot: float=0.02,
                 search_iter:int = 0,
                 number_of_samples = None):
        super().__init__("DeepFool", model)
        self.steps = steps
        self.overshoot = overshoot
        self.supported_mode = ['default']
        self.search_iter = search_iter
        self.number_of_samples = number_of_samples
        self.fool_checker = 0
        self.number_of_iterations = 0

    def forward(self, images, labels, return_target_labels=False):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        batch_size = len(images)
        correct = torch.tensor([True]*batch_size)
        target_labels = labels.clone().detach().to(self.device)
        curr_steps = 0

        adv_images = []
        for idx in range(batch_size):
            image = images[idx:idx+1].clone().detach()
            adv_images.append(image)

        while (True in correct) and (curr_steps < self.steps):
            for idx in range(batch_size):
                if not correct[idx]: continue
                early_stop, pre, adv_image = self._forward_indiv(adv_images[idx], labels[idx])
                adv_images[idx] = adv_image
                target_labels[idx] = pre
                if early_stop:
                    correct[idx] = False
            curr_steps += 1
        
        self.number_of_iterations = curr_steps

        adv_images = torch.cat(adv_images).detach()

        if return_target_labels:

            return adv_images, target_labels

        return adv_images

    def _forward_indiv(self, image, label):
        image.requires_grad = True
        fs = self.model(image)[0]
        _, pre = torch.max(fs, dim=0)
        if pre != label:
            return (True, pre, image)

        ws = self._construct_jacobian_parallel(fs, image)
        image = image.detach()

        f_0 = fs[label]
        w_0 = ws[label]

        wrong_classes = [i for i in range(len(fs)) if i != label]
        f_k = fs[wrong_classes]
        w_k = ws[wrong_classes]

        f_prime = f_k - f_0
        w_prime = w_k - w_0
        value = torch.abs(f_prime) \
                / torch.norm(nn.Flatten()(w_prime), p=2, dim=1)
        _, hat_L = torch.min(value, 0)

        delta = (torch.abs(f_prime[hat_L])*w_prime[hat_L] \
                 / (torch.norm(w_prime[hat_L], p=2)**2))

        target_label = hat_L if hat_L < label else hat_L+1

        adv_image = image + (1+self.overshoot)*delta
        adv_image = torch.clamp(adv_image, min=0, max=1).detach()
        return (False, target_label, adv_image)

    def _construct_jacobian(self, y, x):
        x_grads = []
        for idx, y_element in enumerate(y):
            if x.grad is not None:
                x.grad.zero_()
            y_element.backward(retain_graph=(False or idx+1 < len(y)))
            x_grads.append(x.grad.clone().detach())
        return torch.stack(x_grads).reshape(*y.shape, *x.shape)
    
    def _construct_jacobian_parallel(self, y, x):
        y = x.repeat(10, 1, 1, 1).detach().requires_grad_()
        out = self.model(y)
        out2 = out.diag().sum()
        out2.backward()
        return y.grad

class SuperDeepFool(Attack):
    def __init__(
        self,
        model,
        steps: int = 100,
        overshoot: float = 0.02,
        search_iter: int = 0,
        number_of_samples=None,
        l_norm: str = "L2",
    ):
        super().__init__("SuperDeepFool", model)
        self.steps = steps
        self.overshoot = overshoot
        self.deepfool = DeepFool(
            model, steps=steps, overshoot=overshoot, search_iter=10
        )
        self._supported_mode = ["default"]
        self.search_iter = search_iter
        self.number_of_samples = number_of_samples
        self.fool_checker = 0
        self.l_norm = l_norm
        self.target_label = None

    def forward(self, images, labels, verbose: bool = True):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        batch_size = len(images)
        correct = torch.tensor([True] * batch_size)
        curr_steps = 0
        r_tot = torch.zeros_like(images)
        adv_images = [
            images[i :: torch.cuda.device_count()].clone().detach().to(device=i)
            for i in range(torch.cuda.device_count())
        ]
        if batch_size % torch.cuda.device_count() != 0:
            adv_images[-1] = (
                images[
                    torch.cuda.device_count()
                    * (batch_size // torch.cuda.device_count()) :
                ]
                .clone()
                .detach()
                .to(self.device)
            )

        while (True in correct) and (curr_steps < self.steps):
            for idx in range(batch_size):
                image = images[idx : idx + 1]
                label = labels[idx : idx + 1]
                r_ = r_tot[idx : idx + 1]
                adv_image = adv_images[idx]

                fs = self.model(adv_image)[0]
                _, pre = torch.max(fs, dim=0)
                if pre != label:
                    correct[idx] = False
                    continue

                adv_image_Deepfool, target_label = self.deepfool(
                    adv_image, label, return_target_labels=True
                )
                r_i = adv_image_Deepfool - image
                adv_image_Deepfool.requires_grad = True
                fs = self.model(adv_image_Deepfool)[0]
                _, pre = torch.max(fs, dim=0)

                if pre == label:
                    pre = target_label
                cost = fs[pre] - fs[label]

                last_grad = torch.autograd.grad(
                    cost, adv_image_Deepfool, retain_graph=False, create_graph=False
                )[0]

                if self.l_norm == "L2":
                    last_grad = last_grad / last_grad.norm()
                    r_ = (
                        r_
                        + (last_grad * (r_i)).sum()
                        * last_grad
                        / (
                            np.linalg.norm(
                                last_grad.detach().cpu().numpy().flatten(), ord=2
                            )
                        )
                        ** 2
                    )

                adv_image = image + r_
                adv_images[idx] = adv_image.detach()
                r_tot[idx] = r_.detach()
                self.target_label = target_label.detach()

            curr_steps += 1

        adv_images = torch.cat(adv_images).detach()
        if self.search_iter > 0:
            if verbose:
                print(f"search iteration for SuperDeepfool -> {self.search_iter}")
            dx = adv_images - images
            dx_l_low, dx_l_high = torch.zeros_like(dx), torch.ones_like(dx)
            for i in range(self.search_iter):
                dx_l = (dx_l_low + dx_l_high) / 2.0
                dx_x = images + dx_l * dx
                dx_y = self.model(dx_x).argmax(-1)
                label_stay = dx_y == labels
                label_change = dx_y != labels
                dx_l_low[label_stay] = dx_l[label_stay]
                dx_l_high[label_change] = dx_l[label_change]
            adv_images = images + dx_l_high * dx
        return adv_images