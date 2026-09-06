# filter1.py - 통합 필터 모음
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

    # ============================================================
    # 1. 기존 필터
    # ============================================================
    if filter_type == 'indigo':
        img[:, :, 0] *= 0.2   # R
        img[:, :, 1] *= 0.4   # G
        img[:, :, 2] *= 1.8   # B

    elif filter_type == 'noise':
        mask = np.random.random((h, w)) < 0.12
        noise_r = np.random.randint(0, 256, (h, w), dtype=np.uint8)
        noise_g = np.random.randint(0, 256, (h, w), dtype=np.uint8)
        noise_b = np.random.randint(0, 256, (h, w), dtype=np.uint8)
        img[mask, 0] = noise_r[mask]
        img[mask, 1] = noise_g[mask]
        img[mask, 2] = noise_b[mask]
        img[:, :, 0] *= (0.85 + np.random.random((h, w)) * 0.3)
        img[:, :, 1] *= (0.85 + np.random.random((h, w)) * 0.3)
        img[:, :, 2] *= (0.85 + np.random.random((h, w)) * 0.3)

    elif filter_type == 'y2k':
        img = (img / 255.0 - 0.5) * 1.3 + 0.5
        img[:, :, 0] = img[:, :, 0] * 1.15 + 0.05
        img[:, :, 1] = img[:, :, 1] * 0.85
        img[:, :, 2] = img[:, :, 2] * 1.35 + 0.05
        shifted = np.roll(img, shift=3, axis=1)
        img[:, :, 0] = img[:, :, 0] * 0.5 + shifted[:, :, 0] * 0.5

    # ============================================================
    # 2. 새로운 필터 7가지
    # ============================================================
    elif filter_type == 'sepia':
        # 세피아 톤 (复古棕褐色)
        # 변환 행렬 적용
        sepia_matrix = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        # (h, w, 3) 행렬 곱셈 (알파 채널은 그대로)
        rgb = img[:, :, :3]
        sepia_rgb = rgb @ sepia_matrix.T
        img[:, :, :3] = np.clip(sepia_rgb, 0, 255)

    elif filter_type == 'grayscale':
        # 흑백 ( luminance 공식 )
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        img[:, :, 0] = gray
        img[:, :, 1] = gray
        img[:, :, 2] = gray

    elif filter_type == 'invert':
        # 색상 반전 (네거티브)
        img[:, :, :3] = 255 - img[:, :, :3]

    elif filter_type == 'high_contrast':
        # 고대비 (대비 1.5배)
        img = (img / 255.0 - 0.5) * 1.5 + 0.5
        img[:, :, :3] = np.clip(img[:, :, :3] * 255, 0, 255)

    elif filter_type == 'warm':
        # 따뜻한 톤 (R +20, G +10, B -20)
        img[:, :, 0] = np.clip(img[:, :, 0] + 20, 0, 255)
        img[:, :, 1] = np.clip(img[:, :, 1] + 10, 0, 255)
        img[:, :, 2] = np.clip(img[:, :, 2] - 20, 0, 255)

    elif filter_type == 'cool':
        # 차가운 톤 (R -20, G +10, B +30)
        img[:, :, 0] = np.clip(img[:, :, 0] - 20, 0, 255)
        img[:, :, 1] = np.clip(img[:, :, 1] + 10, 0, 255)
        img[:, :, 2] = np.clip(img[:, :, 2] + 30, 0, 255)

    elif filter_type == 'pixelate':
        # 간단한 픽셀화 (8x8 블록 평균)
        block_size = 8
        # 이미지를 블록 단위로 축소했다가 확대
        h_small = h // block_size
        w_small = w // block_size
        # 블록 평균 계산
        img_small = img[:h_small*block_size, :w_small*block_size].reshape(h_small, block_size, w_small, block_size, -1)
        img_small = img_small.mean(axis=(1, 3))  # (h_small, w_small, 4)
        # 확대 (nearest neighbor)
        img_pixelated = np.repeat(np.repeat(img_small, block_size, axis=0), block_size, axis=1)
        # 원본 크기에 맞춤 (나머지 영역은 원본 유지)
        img[:h_small*block_size, :w_small*block_size] = img_pixelated
        # 알파 채널 유지 (블록 평균으로 인해 알파도 변할 수 있지만, 그대로 둠)

    # ============================================================
    # 3. 결과 반환
    # ============================================================
    return np.clip(img, 0, 255).astype(np.uint8)