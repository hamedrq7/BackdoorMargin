import torch
import torch.nn as nn

from attacks.attack import Attack

# img.paste(self.trigger_img, (self.img_width - self.trigger_size, self.img_height - self.trigger_size))

class ConDeepFool(Attack):
    r"""
    Constrainted Deep Fool In Pixel Space
    """
    def __init__(self, model,
                 mask: torch.Tensor, 
                 steps: int=100,
                 overshoot: float=0.02,
                 search_iter:int = 0,
                 number_of_samples = None):
        super().__init__("ConDeepFool", model)
        self.mask = mask.to(self.device)
        self.steps = steps
        self.overshoot = overshoot
        self.supported_mode = ['default']
        self.search_iter = search_iter
        self.number_of_samples = number_of_samples
        self.fool_checker = 0
        self.number_of_iterations = 0

    def forward(self, images, labels, ):
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
        deltas = []
        for idx in range(batch_size):
            image = images[idx:idx+1].clone().detach()
            adv_images.append(image)
            deltas.append(torch.zeros_like(image))

        while (True in correct) and (curr_steps < self.steps):
            for idx in range(batch_size):
                if not correct[idx]: continue
                early_stop, pre, adv_image, delta = self._forward_indiv(adv_images[idx], labels[idx])
                adv_images[idx] = adv_image
                target_labels[idx] = pre
                deltas[idx] += delta
                if early_stop:
                    correct[idx] = False
            curr_steps += 1
        
        self.number_of_iterations = curr_steps

        adv_images = torch.cat(adv_images).detach()
        deltas = torch.cat(deltas).detach()

        return adv_images, target_labels, deltas

    def _forward_indiv(self, image, label):
        image.requires_grad = True
        fs = self.model(image)[0]
        _, pre = torch.max(fs, dim=0)
        if pre != label:
            return (True, pre, image, torch.zeros_like(image))

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

        adv_image = image + (1+self.overshoot)*delta
        adv_image = torch.clamp(adv_image, min=0, max=1).detach()
        
        adv_fs = self.model(adv_image)[0]
        _, adv_pre = torch.max(adv_fs, dim=0)
        early_stop = adv_pre != label
        return early_stop, adv_pre, adv_image, (1+self.overshoot)*delta

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

        # C, T, H, W = 1, 5, 28, 28        
        # mask = torch.zeros(C, H, W)
        # mask[:, H - T:H, W - T:W] = 1.

        return y.grad * self.mask