import sys
sys.path.append('D:/projectMoeumZip/grade4/인공지능/git_giraffe/2025AI_Giraffe/stylegan3')

import torch
import numpy as np
import PIL.Image
from typing import List, Optional
import dnnlib
import legacy


class StyleGAN3StyleMixer:
    def __init__(self, giraffe_model_path: str, human_model_path: str, device: str = 'cpu'):
        self.device = device
        self.giraffe_G = self.load_model(giraffe_model_path)
        self.human_G = self.load_model(human_model_path)
        self.z_dim = self.giraffe_G.z_dim
        self.w_dim = self.giraffe_G.w_dim
        print(f"모델 로드 완료: z_dim={self.z_dim}, w_dim={self.w_dim}")

    def load_model(self, model_path: str):
        print(f"모델 로드 중: {model_path}")
        with open(model_path, 'rb') as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(self.device)
        return G

    def generate_w_codes(self, num_samples: int = 1, seed: Optional[int] = None) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        z = torch.randn(num_samples, self.z_dim, device=self.device)
        w = self.giraffe_G.mapping(z, None)
        return w

    def style_mixing(self, w_giraffe, w_human, mixing_layers: List[int], resolution: int = 512) -> PIL.Image.Image:
        num_layers = self.giraffe_G.synthesis.num_ws
        
        # 조건부 확장
        if w_giraffe.ndim == 2:
            w_giraffe_full = w_giraffe.unsqueeze(1).expand(-1, num_layers, -1)
        else:
            w_giraffe_full = w_giraffe

        if w_human.ndim == 2:
            w_human_full = w_human.unsqueeze(1).expand(-1, num_layers, -1)
        else:
            w_human_full = w_human

        w_mixed = w_giraffe_full.clone()
        for layer_idx in mixing_layers:
            if layer_idx < num_layers:
                w_mixed[:, layer_idx] = w_human_full[:, layer_idx]

        with torch.no_grad():
            img = self.giraffe_G.synthesis(w_mixed, noise_mode='const')

        return self._postprocess_image(img, resolution)

    def cross_domain_mixing(self, base_model: str, w_base, w_style, mixing_layers: List[int], resolution: int = 512) -> PIL.Image.Image:
        G = self.giraffe_G if base_model == 'giraffe' else self.human_G
        num_layers = G.synthesis.num_ws
        w_base_full = w_base.unsqueeze(1).expand(-1, num_layers, -1)
        w_style_full = w_style.unsqueeze(1).expand(-1, num_layers, -1)

        w_mixed = w_base_full.clone()
        for layer_idx in mixing_layers:
            if layer_idx < num_layers:
                w_mixed[:, layer_idx] = w_style_full[:, layer_idx]

        with torch.no_grad():
            img = G.synthesis(w_mixed, noise_mode='const')

        return self._postprocess_image(img, resolution)

    def progressive_mixing(self, w_source, w_target, num_steps: int = 8, resolution: int = 512) -> List[PIL.Image.Image]:
        images = []
        num_layers = self.giraffe_G.synthesis.num_ws
        w_source_full = w_source.unsqueeze(1).expand(-1, num_layers, -1)
        w_target_full = w_target.unsqueeze(1).expand(-1, num_layers, -1)

        for step in range(num_steps):
            alpha = step / (num_steps - 1)
            mixing_layers = list(range(int(alpha * num_layers)))

            w_mixed = w_source_full.clone()
            for layer_idx in mixing_layers:
                w_mixed[:, layer_idx] = w_target_full[:, layer_idx]

            with torch.no_grad():
                img = self.giraffe_G.synthesis(w_mixed, noise_mode='const')

            images.append(self._postprocess_image(img, resolution))
        return images

    def _postprocess_image(self, img: torch.Tensor, resolution: int) -> PIL.Image.Image:
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = img[0].cpu().numpy()
        if img.shape[:2] != (resolution, resolution):
            img = np.array(PIL.Image.fromarray(img).resize((resolution, resolution), PIL.Image.LANCZOS))
        return PIL.Image.fromarray(img)

    def save_image(self, img: PIL.Image.Image, path: str):
        img.save(path)
        print(f"이미지 저장됨: {path}")


def main():
    giraffe_model_path = "giraffe_model.pkl"
    human_model_path = "human_model.pkl"
    mixer = StyleGAN3StyleMixer(giraffe_model_path, human_model_path, device='cpu')

    w_giraffe = mixer.generate_w_codes(1, seed=42)
    w_human = mixer.generate_w_codes(1, seed=123)

    coarse_layers = [0, 1, 2, 3]
    mixed_img1 = mixer.style_mixing(w_giraffe, w_human, coarse_layers)
    mixer.save_image(mixed_img1, "style_mix_coarse.png")

    fine_layers = [8, 9, 10, 11, 12, 13]
    mixed_img2 = mixer.style_mixing(w_giraffe, w_human, fine_layers)
    mixer.save_image(mixed_img2, "style_mix_fine.png")

    mid_layers = [4, 5, 6, 7]
    mixed_img3 = mixer.style_mixing(w_giraffe, w_human, mid_layers)
    mixer.save_image(mixed_img3, "style_mix_mid.png")

    cross_img = mixer.cross_domain_mixing('giraffe', w_giraffe, w_human, [4, 5, 6, 7, 8])
    mixer.save_image(cross_img, "cross_domain_mix.png")

    progressive_imgs = mixer.progressive_mixing(w_giraffe, w_human, num_steps=8)
    for i, img in enumerate(progressive_imgs):
        mixer.save_image(img, f"progressive_mix_{i:02d}.png")

    print("Style mixing 완료!")


if __name__ == "__main__":
    main()
