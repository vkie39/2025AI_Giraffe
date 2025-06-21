import sys
sys.path.append('D:/projectMoeumZip/grade4/인공지능/git_giraffe/2025AI_Giraffe/stylegan3')  # 여기에 dnnlib 폴더가 있어야 함

import torch
import torch.nn.functional as F
import numpy as np
import PIL.Image
from typing import List, Optional, Tuple
import pickle
import dnnlib
import legacy


class StyleGAN3StyleMixer:
    def __init__(self, giraffe_model_path: str, human_model_path: str, device: str = 'cuda'):
        """
        StyleGAN3 Style Mixing 클래스
        
        Args:
            giraffe_model_path: 기린 학습 모델 경로 (.pkl)
            human_model_path: 사람 얼굴 학습 모델 경로 (.pkl)
            device: 'cuda' 또는 'cpu'
        """
        self.device = device
        
        # 모델 로드
        self.giraffe_G = self.load_model(giraffe_model_path)
        self.human_G = self.load_model(human_model_path)
        
        # 잠재 공간 차원 확인
        self.z_dim = self.giraffe_G.z_dim
        self.w_dim = self.giraffe_G.w_dim
        
        print(f"모델 로드 완료: z_dim={self.z_dim}, w_dim={self.w_dim}")
    
    def load_model(self, model_path: str):
        """StyleGAN3 모델 로드"""
        print(f"모델 로드 중: {model_path}")
        with open(model_path, 'rb') as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(self.device)
        return G
    
    def generate_w_codes(self, num_samples: int = 1, seed: Optional[int] = None) -> torch.Tensor:
        """랜덤 W 코드 생성"""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        z = torch.randn(num_samples, self.z_dim, device=self.device)
        w = self.giraffe_G.mapping(z, None)  # 기린 모델의 매핑 네트워크 사용
        return w
    
    def style_mixing(self, 
                    w_giraffe: torch.Tensor,
                    w_human: torch.Tensor,
                    mixing_layers: List[int],
                    resolution: int = 512) -> PIL.Image.Image:
        """
        Style mixing 수행
        
        Args:
            w_giraffe: 기린 모델의 W 코드
            w_human: 사람 모델의 W 코드  
            mixing_layers: 혼합할 레이어 인덱스 리스트 (예: [0,1,2,3])
            resolution: 출력 해상도
            
        Returns:
            혼합된 이미지
        """
        # W 코드를 확장 (모든 레이어에 대해)
        num_layers = self.giraffe_G.synthesis.num_ws
        w_giraffe_full = w_giraffe.repeat(1, num_layers, 1)
        w_human_full = w_human.repeat(1, num_layers, 1)
        
        # 지정된 레이어에서 스타일 혼합
        w_mixed = w_giraffe_full.clone()
        for layer_idx in mixing_layers:
            if layer_idx < num_layers:
                w_mixed[:, layer_idx] = w_human_full[:, layer_idx]
        
        # 기린 모델로 이미지 생성 (기본 구조는 기린 모델 사용)
        with torch.no_grad():
            img = self.giraffe_G.synthesis(w_mixed, noise_mode='const')
        
        # 이미지 후처리
        img = self._postprocess_image(img, resolution)
        return img
    
    def cross_domain_mixing(self,
                           base_model: str,
                           w_base: torch.Tensor,
                           w_style: torch.Tensor,
                           mixing_layers: List[int],
                           resolution: int = 512) -> PIL.Image.Image:
        """
        도메인 간 스타일 혼합
        
        Args:
            base_model: 'giraffe' 또는 'human' - 기본 구조를 제공할 모델
            w_base: 기본 모델의 W 코드
            w_style: 스타일을 가져올 W 코드
            mixing_layers: 혼합할 레이어 인덱스
            resolution: 출력 해상도
        """
        G = self.giraffe_G if base_model == 'giraffe' else self.human_G
        
        num_layers = G.synthesis.num_ws
        w_base_full = w_base.repeat(1, num_layers, 1)
        w_style_full = w_style.repeat(1, num_layers, 1)
        
        w_mixed = w_base_full.clone()
        for layer_idx in mixing_layers:
            if layer_idx < num_layers:
                w_mixed[:, layer_idx] = w_style_full[:, layer_idx]
        
        with torch.no_grad():
            img = G.synthesis(w_mixed, noise_mode='const')
        
        img = self._postprocess_image(img, resolution)
        return img
    
    def progressive_mixing(self,
                          w_source: torch.Tensor,
                          w_target: torch.Tensor,
                          num_steps: int = 8,
                          resolution: int = 512) -> List[PIL.Image.Image]:
        """
        점진적 스타일 혼합 (애니메이션용)
        
        Args:
            w_source: 시작 W 코드
            w_target: 목표 W 코드
            num_steps: 중간 단계 수
            resolution: 출력 해상도
            
        Returns:
            중간 단계 이미지들의 리스트
        """
        images = []
        num_layers = self.giraffe_G.synthesis.num_ws
        
        for step in range(num_steps):
            alpha = step / (num_steps - 1)
            
            # 점진적으로 더 많은 레이어를 혼합
            mixing_layers = list(range(int(alpha * num_layers)))
            
            w_source_full = w_source.repeat(1, num_layers, 1)
            w_target_full = w_target.repeat(1, num_layers, 1)
            
            w_mixed = w_source_full.clone()
            for layer_idx in mixing_layers:
                w_mixed[:, layer_idx] = w_target_full[:, layer_idx]
            
            with torch.no_grad():
                img = self.giraffe_G.synthesis(w_mixed, noise_mode='const')
            
            img = self._postprocess_image(img, resolution)
            images.append(img)
        
        return images
    
    def _postprocess_image(self, img: torch.Tensor, resolution: int) -> PIL.Image.Image:
        """이미지 후처리"""
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = img[0].cpu().numpy()
        
        if img.shape[:2] != (resolution, resolution):
            img = np.array(PIL.Image.fromarray(img).resize((resolution, resolution), PIL.Image.LANCZOS))
        
        return PIL.Image.fromarray(img)
    
    def save_image(self, img: PIL.Image.Image, path: str):
        """이미지 저장"""
        img.save(path)
        print(f"이미지 저장됨: {path}")

# 사용 예제
def main():
    # 모델 경로 설정
    giraffe_model_path = "giraffe_model.pkl"
    human_model_path = "human_model.pkl"
    
    # StyleMixer 초기화
    mixer = StyleGAN3StyleMixer(giraffe_model_path, human_model_path)
    
    # 랜덤 W 코드 생성
    w_giraffe = mixer.generate_w_codes(1, seed=42)
    w_human = mixer.generate_w_codes(1, seed=123)
    
    # 1. 기본 스타일 혼합 (저해상도 레이어만)
    coarse_layers = [0, 1, 2, 3]  # 거친 특징 (전체적인 구조)
    mixed_img1 = mixer.style_mixing(w_giraffe, w_human, coarse_layers)
    mixer.save_image(mixed_img1, "style_mix_coarse.png")
    
    # 2. 세부 스타일 혼합 (고해상도 레이어)
    fine_layers = [8, 9, 10, 11, 12, 13, 14, 15]  # 세부 특징 (텍스처, 색상)
    mixed_img2 = mixer.style_mixing(w_giraffe, w_human, fine_layers)
    mixer.save_image(mixed_img2, "style_mix_fine.png")
    
    # 3. 중간 레이어 혼합
    mid_layers = [4, 5, 6, 7]  # 중간 특징
    mixed_img3 = mixer.style_mixing(w_giraffe, w_human, mid_layers)
    mixer.save_image(mixed_img3, "style_mix_mid.png")
    
    # 4. 도메인 간 혼합 (기린 구조 + 사람 스타일)
    cross_img = mixer.cross_domain_mixing('giraffe', w_giraffe, w_human, [4,5,6,7,8])
    mixer.save_image(cross_img, "cross_domain_mix.png")
    
    # 5. 점진적 혼합 (애니메이션)
    progressive_imgs = mixer.progressive_mixing(w_giraffe, w_human, num_steps=8)
    for i, img in enumerate(progressive_imgs):
        mixer.save_image(img, f"progressive_mix_{i:02d}.png")
    
    print("Style mixing 완료!")

if __name__ == "__main__":
    main()
