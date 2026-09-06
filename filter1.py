# filter1.py - 통합 필터 모음 (노이즈 개선 + 렌즈 효과)
import numpy as np

def apply_filter(image_data, filter_type):
    """
    image_data: np.ndarray (height, width, 4) RGBA, dtype=uint8
    filter_type: 지원하는 필터 이름
    반환: 처리된 np.ndarray (height, width, 4) dtype=uint8
    """
    if filter_type == 'original':
        return image_data

    img = image_data.astype(np.float32)
    h, w, _ = img.shape
    
    # 알파 채널 보존
    alpha = img[:, :, 3:4]

    # ============================================================
    # 1. 기존 필터
    # ============================================================
    if filter_type == 'indigo':
        img[:, :, 0] *= 0.2
        img[:, :, 1] *= 0.4
        img[:, :, 2] *= 1.8

    elif filter_type == 'noise':
        # ⭐ 흑백 그레인 노이즈 (잡티 제거용)
        # 가우시안 노이즈 + 소금-후추 노이즈 혼합
        grain_strength = 37  # 강도 조절 (0~50)
        
        # 1) 가우시안 노이즈 (미세한 밝기 변화)
        gaussian_noise = np.random.normal(0, grain_strength/3, (h, w, 3))
        
        # 2) 소금-후추 노이즈 (흰색/검은색 점)
        salt_pepper_mask = np.random.random((h, w)) < 0.02  # 2% 확률
        salt_pepper_noise = np.zeros((h, w, 3))
        salt_pepper_noise[salt_pepper_mask] = np.random.choice([0, 255], size=(np.sum(salt_pepper_mask), 3))
        
        # 노이즈 합치기 (흑백으로 통일)
        noise = gaussian_noise + salt_pepper_noise
        
        # 모든 채널에 같은 노이즈 적용 (흑백 그레인)
        gray_noise = np.mean(noise, axis=2, keepdims=True)
        img[:, :, :3] = np.clip(img[:, :, :3] + gray_noise, 0, 255)

    elif filter_type == 'y2k':
        rgb = img[:, :, :3]
        rgb = (rgb / 255.0 - 0.5) * 1.3 + 0.5
        rgb[:, :, 0] = rgb[:, :, 0] * 1.15 + 0.05
        rgb[:, :, 1] = rgb[:, :, 1] * 0.85
        rgb[:, :, 2] = rgb[:, :, 2] * 1.35 + 0.05
        shifted = np.roll(rgb, shift=3, axis=1)
        rgb[:, :, 0] = rgb[:, :, 0] * 0.5 + shifted[:, :, 0] * 0.5
        rgb = np.clip(rgb * 255, 0, 255)
        img[:, :, :3] = rgb

    # ============================================================
    # 2. 색상 필터
    # ============================================================
    elif filter_type == 'sepia':
        sepia_matrix = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        rgb = img[:, :, :3]
        sepia_rgb = rgb @ sepia_matrix.T
        img[:, :, :3] = np.clip(sepia_rgb, 0, 255)

    elif filter_type == 'grayscale':
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        img[:, :, 0] = gray
        img[:, :, 1] = gray
        img[:, :, 2] = gray

    elif filter_type == 'invert':
        img[:, :, :3] = 255 - img[:, :, :3]

    elif filter_type == 'high_contrast':
        rgb = img[:, :, :3]
        rgb = (rgb / 255.0 - 0.5) * 1.5 + 0.5
        rgb = np.clip(rgb * 255, 0, 255)
        img[:, :, :3] = rgb

    elif filter_type == 'warm':
        img[:, :, 0] = np.clip(img[:, :, 0] + 20, 0, 255)
        img[:, :, 1] = np.clip(img[:, :, 1] + 10, 0, 255)
        img[:, :, 2] = np.clip(img[:, :, 2] - 20, 0, 255)

    elif filter_type == 'cool':
        img[:, :, 0] = np.clip(img[:, :, 0] - 20, 0, 255)
        img[:, :, 1] = np.clip(img[:, :, 1] + 10, 0, 255)
        img[:, :, 2] = np.clip(img[:, :, 2] + 30, 0, 255)

    elif filter_type == 'pixelate':
        block_size = 8
        h_small = h // block_size
        w_small = w // block_size
        rgb = img[:h_small*block_size, :w_small*block_size, :3]
        rgb_small = rgb.reshape(h_small, block_size, w_small, block_size, 3)
        rgb_small = rgb_small.mean(axis=(1, 3))
        rgb_pixelated = np.repeat(np.repeat(rgb_small, block_size, axis=0), block_size, axis=1)
        img[:h_small*block_size, :w_small*block_size, :3] = rgb_pixelated

    # ============================================================
    # 3. 렌즈 효과
    # ============================================================
    elif filter_type == 'fisheye':
        img = apply_fisheye(img, strength=1.8)

    elif filter_type == 'bulge':
        img = apply_bulge(img, strength=1.5)

    elif filter_type == 'pinch':
        img = apply_bulge(img, strength=0.6)

    # ============================================================
    # 4. 알파 채널 복원 및 결과 반환
    # ============================================================
    img[:, :, 3:4] = alpha
    return np.clip(img, 0, 255).astype(np.uint8)


# ============================================================
# 렌즈 효과 헬퍼 함수들
# ============================================================

def apply_fisheye(img, strength=1.8):
    """
    어안 효과 (Fisheye)
    - 중심에서 바깥으로 갈수록 이미지가 휘어짐
    - strength: 왜곡 강도 (1.0~2.5)
    """
    h, w, _ = img.shape
    cy, cx = h // 2, w // 2
    
    y, x = np.ogrid[:h, :w]
    y_norm = (y - cy) / cy
    x_norm = (x - cx) / cx
    
    r = np.sqrt(x_norm**2 + y_norm**2)
    theta = np.arctan2(y_norm, x_norm)
    
    r_new = r / (1 + strength * r * r)
    
    x_new = cx + r_new * np.cos(theta) * cx
    y_new = cy + r_new * np.sin(theta) * cy
    
    return remap_image(img, x_new, y_new)


def apply_bulge(img, strength=1.5):
    """
    볼록/오목 렌즈 효과
    - strength > 1: 볼록 (확대)
    - strength < 1: 오목 (축소)
    """
    h, w, _ = img.shape
    cy, cx = h // 2, w // 2
    
    y, x = np.ogrid[:h, :w]
    y_norm = (y - cy) / cy
    x_norm = (x - cx) / cx
    
    r = np.sqrt(x_norm**2 + y_norm**2)
    theta = np.arctan2(y_norm, x_norm)
    
    r_new = np.power(r, 1.0 / strength)
    r_new = np.clip(r_new, 0, 1)
    
    x_new = cx + r_new * np.cos(theta) * cx
    y_new = cy + r_new * np.sin(theta) * cy
    
    return remap_image(img, x_new, y_new)


def remap_image(img, x_map, y_map):
    """
    Bilinear interpolation을 이용한 이미지 리매핑
    """
    h, w, c = img.shape
    
    x_map = x_map.astype(np.float32)
    y_map = y_map.astype(np.float32)
    
    x_map = np.clip(x_map, 0, w - 1)
    y_map = np.clip(y_map, 0, h - 1)
    
    x0 = np.floor(x_map).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y0 = np.floor(y_map).astype(np.int32)
    y1 = np.minimum(y0 + 1, h - 1)
    
    wa = (x1 - x_map) * (y1 - y_map)
    wb = (x_map - x0) * (y1 - y_map)
    wc = (x1 - x_map) * (y_map - y0)
    wd = (x_map - x0) * (y_map - y0)
    
    Ia = img[y0, x0]
    Ib = img[y0, x1]
    Ic = img[y1, x0]
    Id = img[y1, x1]
    
    result = (Ia * wa[:, :, np.newaxis] +
              Ib * wb[:, :, np.newaxis] +
              Ic * wc[:, :, np.newaxis] +
              Id * wd[:, :, np.newaxis])
    
    return result