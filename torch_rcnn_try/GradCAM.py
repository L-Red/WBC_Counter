import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.feature = None
        self.gradient = None
        self.model.eval()
        self.handlers = []
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        self.feature = output[0].detach()
        self.feature_numpy = self.feature.cpu().numpy()  # copy to numpy here

    def _get_grads_hook(self, module, input_grad, output_grad):
        self.gradient = output_grad[0].detach()

        # print the min, max, and mean of the gradients
        print('Gradients: min: ', self.gradient.min(), ' max: ', self.gradient.max(), ' mean: ', self.gradient.mean())

        self.gradient_numpy = self.gradient.cpu().numpy()  # copy to numpy here

        # compute the global average pooled gradients
        self.gradient = self.gradient.mean(dim=[0, 2, 3], keepdim=True)

        # print the min, max, and mean of the pooled gradients
        print('Pooled gradients: min: ', self.gradient.min(), ' max: ', self.gradient.max(), ' mean: ',
              self.gradient.mean())

    def _register_hook(self):
        for name, module in self.model.named_modules():
            if name == 'layer4':
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_full_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, data, class_idx=None, retain_graph=False):
        self.model.zero_grad()
        output = self.model(data)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        target = output[:, class_idx]
        target.backward(retain_graph=retain_graph)

        weights = np.mean(self.gradient_numpy, axis=(1, 2))  # use the numpy copy

        cam = np.zeros(self.feature_numpy.shape[1:], dtype=np.float32)  # use the numpy copy

        for i, w in enumerate(weights):
            cam += w * self.feature_numpy[i, :, :]  # use the numpy copy
        # print the min, max, and mean of the cam
        print('CAM: min: ', np.min(cam), ' max: ', np.max(cam), ' mean: ', np.mean(cam))

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, data.shape[2:])
        cam = cam - np.min(cam)
        if np.max(cam) != 0:
            cam = cam / (np.max(cam))
        self.remove_handlers()

        return cam

    def draw(self, heatmap, image):
        print(f'heatmap shape: {heatmap.shape}')
        # move image to cpu, convert to numpy and change the datatype
        image = (image * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        heatmap = (heatmap * 255).astype(np.uint8)

        heatmap = np.expand_dims(heatmap, axis=-1)

        # apply color map to the heatmap
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # superimpose the heatmap on the image
        superimposed_img = heatmap + image
        superimposed_img = superimposed_img.astype(np.float32)

        # normalize if maximum is not zero
        if superimposed_img.max() != 0:
            superimposed_img = (superimposed_img / superimposed_img.max())

        # convert to uint8 and move back to torch tensor
        superimposed_img = torch.from_numpy(superimposed_img)

        return superimposed_img.permute(2, 0, 1)  # change back to (channel, height, width) format

