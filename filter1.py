# filters.py
import numpy as np

def apply_filter(image_data, filter_type):
    """
    image_data: np.ndarray 형태 (height, width, 4) RGBA, dtype=uint8
    filter_type: 'original', 'indigo', 'noise', 'y2k'
    반환: 처리된 np.ndarray (height, width, 4) dtype=uint8
    """
    if filter_type == 'original':
        return image_data

    # float32로 변환하여 연산 (오버플로 방지)
    img = image_data.astype(np.float32)
    h, w, _ = img.shape

    if filter_type == 'indigo':
        # 남색 계열: R,G 낮추고 B 극대화
        img[:, :, 0] = img[:, :, 0] * 0.2   # R
        img[:, :, 1] = img[:, :, 1] * 0.4   # G
        img[:, :, 2] = img[:, :, 2] * 1.8   # B

    elif filter_type == 'noise':
        # 12% 확률로 랜덤 컬러 노이즈 삽입 (TV 노이즈 느낌)
        mask = np.random.random((h, w)) < 0.12
        noise_r = np.random.randint(0, 256, (h, w), dtype=np.uint8)
        noise_g = np.random.randint(0, 256, (h, w), dtype=np.uint8)
        noise_b = np.random.randint(0, 256, (h, w), dtype=np.uint8)
        # mask가 True인 위치에 노이즈 덮어쓰기
        img[mask, 0] = noise_r[mask]
        img[mask, 1] = noise_g[mask]
        img[mask, 2] = noise_b[mask]
        # 노이즈가 아닌 픽셀도 미세하게 흔들기 (빈티지)
        img[:, :, 0] = img[:, :, 0] * (0.85 + np.random.random((h, w)) * 0.3)
        img[:, :, 1] = img[:, :, 1] * (0.85 + np.random.random((h, w)) * 0.3)
        img[:, :, 2] = img[:, :, 2] * (0.85 + np.random.random((h, w)) * 0.3)

    elif filter_type == 'y2k':
        # 1. 콘트라스트 부스트 (1.3배)
        img = (img / 255.0 - 0.5) * 1.3 + 0.5
        # 2. 색상 왜곡 (핑크/시안 쉬프트)
        img[:, :, 0] = img[:, :, 0] * 1.15 + 0.05  # R (핑크)
        img[:, :, 1] = img[:, :, 1] * 0.85         # G (약화)
        img[:, :, 2] = img[:, :, 2] * 1.35 + 0.05  # B (시안)
        # 3. 크로마틱 어버레이션 (Red 채널을 오른쪽으로 3픽셀 밀기)
        shifted = np.roll(img, shift=3, axis=1)    # 가로로 3픽셀 이동
        img[:, :, 0] = img[:, :, 0] * 0.5 + shifted[:, :, 0] * 0.5

    # 0~255 범위로 클리핑 후 uint8 변환
    return np.clip(img, 0, 255).astype(np.uint8)