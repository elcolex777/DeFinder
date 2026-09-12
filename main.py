import os
import os.path
import io
import uuid
import json
import base64
import asyncio
from typing import List, Optional
import torch
import open_clip
import numpy as np
import faiss
from PIL import Image
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, UUID4
from ultralytics import SAM, FastSAM

app = FastAPI(title="MobileSAM & CLIP FAISS Indexing Service")

# --- ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ ---
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading MobileSAM model on {device}...")
#sam_model = SAM("mobile_sam.pt")
sam_model = FastSAM("FastSAM-s.pt")

print(f"Loading CLIP model (ViT-B-32) on {device}...")
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_e16')
clip_model = clip_model.to(device)
clip_model.eval()
print(f"Loaded CLIP model (ViT-B-32) on {device}...")

DIMENSION = 512


# --- PYDANTIC СХЕМЫ ДАННЫХ ---
class MaskPredictionRequest(BaseModel):
    image_base64: str = Field(..., description="Изображение в формате Base64")
    conf_threshold: float = Field(0.25, description="Порог уверенности (conf) для MobileSAM")
    iou_threshold: float = Field(0.7, description="Порог IoU для NMS в MobileSAM")
    imgsz: Optional[int] = Field(320, description="Размер изображения для инференса MobileSAM")
    min_width: float = Field(10.0, description="Минимальная ширина объекта (полигона) в пикселях")

class MaskPredictionResponse(BaseModel):
    masks: List[List[int]]
    mask_width: int
    mask_height: int

class IndexMasksRequest(BaseModel):
    user_id: UUID4 = Field(..., description="GUID пользователя")
    image_base64: str = Field(..., description="Изображение в формате Base64")
    masks: List[List[int]] = Field(..., description="Массив RLE-масок из predict_masks")

class SearchRequest(BaseModel):
    user_id: UUID4 = Field(..., description="GUID пользователя")
    image_base64: str = Field(..., description="Изображение для поиска в Base64")

class SearchResultItem(BaseModel):
    image_path: str
    crop_path: Optional[str] = Field(None, description="Путь к файлу кропа маски")
    center: Optional[dict] = Field(None, description="Координаты центра маски {x, y}")
    score: float = Field(..., description="Косинусное сходство (от -1 до 1, чем выше — тем ближе)")

class SearchResponse(BaseModel):
    results: List[SearchResultItem]


# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
def decode_base64_image(base64_str: str) -> Image.Image:
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Некорректный base64: {str(e)}")

def get_user_paths(user_id: str):
    base_dir = f"data/{user_id}"
    index_dir = f"{base_dir}/index"
    images_dir = f"{base_dir}/images"
    os.makedirs(index_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    index_path = f"{index_dir}/faiss.index"
    metadata_path = f"{index_dir}/metadata.json"
    return index_path, metadata_path, images_dir

def load_or_create_index(index_path: str):
    if os.path.exists(index_path):
        return faiss.read_index(index_path)
    else:
        # Используем IndexFlatIP для косинусного сходства (при условии нормализации векторов)
        return faiss.IndexFlatIP(DIMENSION)

def load_metadata(metadata_path: str) -> dict:
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_metadata(metadata_path: str, data: dict):
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def extract_clip_embedding(image: Image.Image) -> np.ndarray:
    """Извлекает нормализованный эмбеддинг из изображения с помощью CLIP."""
    with torch.no_grad():
        # Препроцессинг и добавление размерности батча [1, C, H, W]
        img_tensor = preprocess(image).unsqueeze(0).to(device)
        # Получаем признаки и убираем градиенты
        image_features = clip_model.encode_image(img_tensor)
        # Переводим в numpy array (форма:)
        embedding = image_features.cpu().numpy().flatten().astype('float32')
        
        # Нормализуем вектор для корректной работы IndexFlatIP (косинусное сходство)
        faiss.normalize_L2(embedding.reshape(1, -1))
        return embedding

def encode_rle2(mask: np.ndarray) -> List[int]:
    """Сжатие бинарной 2D-маски (0 и 1) в формат RLE (COCO-style)."""
    pixels = mask.flatten()
    # Добавляем нули по краям, чтобы корректно зафиксировать изменения на границах массива
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return runs.tolist()

def encode_rle(mask: np.ndarray) -> List[int]:
    """Быстрое векторизованное сжатие бинарной 2D-маски (0 и 1) в формат RLE."""
    pixels = mask.flatten()
    
    # Находим позиции, где значения пикселей меняются (с 0 на 1 или с 1 на 0)
    changes = np.diff(pixels)
    change_indices = np.where(changes != 0)[0] + 1
    
    # Формируем длины серий (разности между индексами переключений)
    if len(change_indices) == 0:
        runs = [len(pixels)]
    else:
        runs = np.diff(np.concatenate(([0], change_indices, [len(pixels)])))
        runs = runs.tolist()
    
    # По стандарту кодирования RLE, массив всегда должен начинаться с подсчета количества нулей.
    # Если маска сразу начинается с единицы, добавляем технический 0 в начало.
    if len(pixels) > 0 and pixels[0] == 1:
        runs.insert(0, 0)
        
    return [int(x) for x in runs]


# --- КОНЕЧНЫЕ ТОЧКИ API ---

# 1. Предсказание масок
@app.post("/api/v1/masks/predict", response_model=MaskPredictionResponse)
async def predict_masks(payload: MaskPredictionRequest):
    image = decode_base64_image(payload.image_base64)
    
    # retina_masks=True интерполирует маски точно в размер входного изображения `image`
    results = await asyncio.to_thread(
        sam_model.predict, 
        image, 
        conf=payload.conf_threshold, 
        iou=payload.iou_threshold, 
        imgsz=payload.imgsz,
        retina_masks=True,
        verbose=False
    )

    output_masks = []
    mask_h, mask_w = 0, 0
    
    if results and len(results) > 0 and results[0].masks is not None:
        masks_data = results[0].masks.data.cpu().numpy().astype(np.uint8)
        
        # Получаем реальные размеры сгенерированных 2D-масок
        if len(masks_data) > 0:
            mask_h, mask_w = masks_data.shape[1], masks_data.shape[2]
        
        if results[0].boxes is not None:
            boxes_data = results[0].boxes.xyxy.cpu().numpy()
            
            for mask, box in zip(masks_data, boxes_data):
                x1, y1, x2, y2 = box
                box_width = x2 - x1
                box_h = y2 - y1
                
                if box_width >= payload.min_width or box_h >= payload.min_width:
                    rle_mask = encode_rle(mask)
                    output_masks.append(rle_mask)
        
    return MaskPredictionResponse(
        masks=output_masks,
        mask_width=mask_w,
        mask_height=mask_h
    )


def decode_rle_and_get_crop_info(rle_runs: List[int], width: int, height: int):
    """
    Разжимает RLE-маску и возвращает:
    - bbox: (x1, y1, x2, y2)
    - center: {"x": ..., "y": ...}
    - mask_2d: бинарный numpy-массив формы (H, W), где 1 - маска, 0 - фон.
    """
    total_pixels = width * height
    if not rle_runs:
        return None

    flat_mask = np.zeros(total_pixels, dtype=np.uint8)
    curr_idx = 0
    val = 0
    
    for length in rle_runs:
        if length > 0:
            if val == 1:
                flat_mask[curr_idx : curr_idx + length] = 1
            curr_idx += length
        val = 1 - val

    mask_2d = flat_mask[:total_pixels].reshape((height, width))
    y_indices, x_indices = np.where(mask_2d == 1)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None

    x1 = int(np.min(x_indices))
    y1 = int(np.min(y_indices))
    x2 = min(width, int(np.max(x_indices)) + 1)
    y2 = min(height, int(np.max(y_indices)) + 1)

    center_x = round(float(np.mean(x_indices)), 2)
    center_y = round(float(np.mean(y_indices)), 2)

    bbox = (x1, y1, x2, y2)
    center = {"x": center_x, "y": center_y}

    return bbox, center, mask_2d


def save_image_to_disk(img: Image.Image, file_path: str):
    """Сохраняет изображение в JPEG или PNG в зависимости от расширения."""
    dir_name = os.path.dirname(file_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".png":
        img.save(file_path, format="PNG")
    else:
        # Для JPEG преобразуем в RGB, если вдруг передан RGBA
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(file_path, format="JPEG", quality=95)

def extract_clip_embeddings(images: List[Image.Image]) -> np.ndarray:
    """Извлекает нормализованные эмбеддинги для списка изображений с помощью CLIP."""
    if not images:
        return np.empty((0, DIMENSION), dtype='float32')
    
    with torch.no_grad():
        img_tensors = torch.stack([preprocess(img) for img in images]).to(device)
        image_features = clip_model.encode_image(img_tensors)
        embeddings = image_features.cpu().numpy().astype('float32')
        faiss.normalize_L2(embeddings)
        return embeddings

def get_bbox_and_center(coords: list, img_width: int, img_height: int) -> Optional[tuple]:
    """
    Вычисляет [x1, y1, x2, y2] для crop и координаты центральной точки (center_x, center_y).
    """
    pts = np.array(coords, dtype=float)
    if pts.size == 0:
        return None

    # Вариант 1: передан массив точек полигона [[x1, y1], [x2, y2], ...]
    if pts.ndim == 2 and pts.shape[1] == 2:
        x1 = float(pts[:, 0].min())
        y1 = float(pts[:, 1].min())
        x2 = float(pts[:, 0].max())
        y2 = float(pts[:, 1].max())
        center_x = float(pts[:, 0].mean())
        center_y = float(pts[:, 1].mean())

    # Вариант 2: передан bbox формата [[x1, y1, x2, y2]]
    elif pts.ndim == 2 and pts.shape[1] == 4 and pts.shape[0] == 1:
        x1, y1, x2, y2 = [float(v) for v in pts[0]]
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

    # Вариант 3: передан плоский bbox формата [x1, y1, x2, y2]
    elif pts.ndim == 1 and len(pts) == 4:
        x1, y1, x2, y2 = [float(v) for v in pts]
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
    else:
        return None

    # Ограничиваем координаты границами кадра для crop
    crop_x1 = max(0, int(x1))
    crop_y1 = max(0, int(y1))
    crop_x2 = min(img_width, int(x2))
    crop_y2 = min(img_height, int(y2))

    if crop_x2 > crop_x1 and crop_y2 > crop_y1:
        bbox = (crop_x1, crop_y1, crop_x2, crop_y2)
        # Округляем координаты центра для компактности
        center = (round(center_x, 2), round(center_y, 2))
        return bbox, center

    return None


def extract_clip_embeddings(images: List[Image.Image]) -> np.ndarray:
    """Извлекает нормализованные эмбеддинги для батча изображений с помощью CLIP."""
    if not images:
        return np.empty((0, DIMENSION), dtype='float32')
    
    with torch.no_grad():
        img_tensors = torch.stack([preprocess(img) for img in images]).to(device)
        image_features = clip_model.encode_image(img_tensors)
        embeddings = image_features.cpu().numpy().astype('float32')
        faiss.normalize_L2(embeddings)
        return embeddings


def extract_clip_embedding(image: Image.Image) -> np.ndarray:
    """Извлекает эмбеддинг одного изображения (для обратной совместимости в поиске)."""
    return extract_clip_embeddings([image])[0]

def save_crop_image(img: Image.Image, path: str):
    img.save(path, format="JPEG", quality=95)


@app.post("/api/v1/masks/index")
async def index_masks(payload: IndexMasksRequest):
    user_str = str(payload.user_id)
    image = decode_base64_image(payload.image_base64)
    img_w, img_h = image.size
    
    index_path, metadata_path, images_dir = get_user_paths(user_str)
    
    # 1. Подготовка директорий и путей
    img_id = str(uuid.uuid4())
    img_filename = f"{img_id}.jpg"
    full_image_path = os.path.join(images_dir, img_filename)
    crops_dir = os.path.join(images_dir, img_id)
    
    await asyncio.to_thread(os.makedirs, crops_dir, exist_ok=True)
    await asyncio.to_thread(save_image_to_disk, image, full_image_path)
    
    # Списки для формируемых объектов
    images_to_save: List[Image.Image] = []
    clip_inputs: List[Image.Image] = []
    crop_paths: List[str] = []
    crop_centers: List[dict] = []
    crop_types: List[str] = []
    crop_orig_indices: List[int] = []

    # Преобразуем картинку в RGBA и массив для вырезания прозрачной маски
    image_rgba = image.convert("RGBA")
    np_rgba = np.array(image_rgba)

    for mask_idx, rle_mask in enumerate(payload.masks):
        res = decode_rle_and_get_crop_info(rle_mask, img_w, img_h)
        if res is None:
            continue
            
        bbox, center, mask_2d = res
        x1, y1, x2, y2 = bbox

        # --- 1. Стандартный прямоугольный кроп ---
        crop_standard = image.crop(bbox)
        images_to_save.append(crop_standard)
        clip_inputs.append(crop_standard)
        crop_paths.append(os.path.join(crops_dir, f"crop_{mask_idx}_standard.jpg"))
        crop_centers.append(center)
        crop_types.append("standard")
        crop_orig_indices.append(mask_idx)

        # --- 2. Расширенный кроп (+20px со всех сторон, кроме упирающихся в край) ---
        pad = 20
        new_x1 = max(0, x1 - pad) if x1 > 0 else 0
        new_y1 = max(0, y1 - pad) if y1 > 0 else 0
        new_x2 = min(img_w, x2 + pad) if x2 < img_w else img_w
        new_y2 = min(img_h, y2 + pad) if y2 < img_h else img_h

        expanded_bbox = (new_x1, new_y1, new_x2, new_y2)
        crop_expanded = image.crop(expanded_bbox)
        images_to_save.append(crop_expanded)
        clip_inputs.append(crop_expanded)
        crop_paths.append(os.path.join(crops_dir, f"crop_{mask_idx}_expanded.jpg"))
        crop_centers.append(center)
        crop_types.append("expanded")
        crop_orig_indices.append(mask_idx)

        # --- 3. Кроп маски с прозрачным фоном (RGBA PNG) ---
        isolated_rgba = np_rgba.copy()
        # Зануляем альфа-канал вне маски
        isolated_rgba[mask_2d == 0, 3] = 0
        
        # Вырезаем область bbox
        isolated_crop_np = isolated_rgba[y1:y2, x1:x2]
        crop_transparent = Image.fromarray(isolated_crop_np, mode="RGBA")
        
        # Для подачи в CLIP (ViT ожидает 3-канальное RGB): накладываем на белый фон
        clip_rgb_transparent = Image.new("RGB", crop_transparent.size, (255, 255, 255))
        clip_rgb_transparent.paste(crop_transparent, mask=crop_transparent.split()[3])

        images_to_save.append(crop_transparent)
        clip_inputs.append(clip_rgb_transparent)
        crop_paths.append(os.path.join(crops_dir, f"crop_{mask_idx}_transparent.png"))
        crop_centers.append(center)
        crop_types.append("transparent")
        crop_orig_indices.append(mask_idx)

    # Fallback, если маски не переданы или пусты
    if not images_to_save:
        crop_standard = image
        images_to_save.append(crop_standard)
        clip_inputs.append(crop_standard)
        crop_paths.append(os.path.join(crops_dir, "crop_0_standard.jpg"))
        crop_centers.append({"x": round(img_w / 2.0, 2), "y": round(img_h / 2.0, 2)})
        crop_types.append("standard")
        crop_orig_indices.append(0)

    # 2. Сохраняем все сформированные изображения на диск
    for img_obj, pth in zip(images_to_save, crop_paths):
        await asyncio.to_thread(save_image_to_disk, img_obj, pth)

    # 3. Извлекаем нормализованные эмбеддинги CLIP для всех вариантов
    embeddings = await asyncio.to_thread(extract_clip_embeddings, clip_inputs)

    # 4. Добавляем векторы в FAISS
    index = await asyncio.to_thread(load_or_create_index, index_path)
    current_vector_id = index.ntotal

    vectors_to_add = embeddings.astype('float32')
    await asyncio.to_thread(index.add, vectors_to_add)
    await asyncio.to_thread(faiss.write_index, index, index_path)

    # 5. Сохраняем метаданные
    metadata = await asyncio.to_thread(load_metadata, metadata_path)
    for i in range(len(images_to_save)):
        vec_id = current_vector_id + i
        metadata[str(vec_id)] = {
            "image_path": full_image_path,
            "crop_path": crop_paths[i],
            "mask_index": crop_orig_indices[i],
            "crop_type": crop_types[i],             # standard, expanded, transparent
            "center": crop_centers[i],
            "masks_count": len(images_to_save)
        }
    await asyncio.to_thread(save_metadata, metadata_path, metadata)

    return {
        "status": "ok",
        "message": f"Успешно проиндексировано объектов: {len(images_to_save)} (по 3 варианта на маску)."
    }


# 3. Поиск по фото
@app.post("/api/v1/masks/search", response_model=SearchResponse)
async def search_image(payload: SearchRequest):
    user_str = str(payload.user_id)
    index_path, metadata_path, _ = get_user_paths(user_str)
    
    # Если у пользователя еще нет индекса
    if not os.path.exists(index_path):
        return SearchResponse(results=[])
    
    # Извлекаем и нормализуем вектор поискового запроса
    query_image = decode_base64_image(payload.image_base64)
    query_vector = await asyncio.to_thread(extract_clip_embedding, query_image)
    query_vector = np.array([query_vector]).astype('float32')
    
    # Читаем индекс и метаданные пользователя с диска
    index = await asyncio.to_thread(load_or_create_index, index_path)
    metadata = await asyncio.to_thread(load_metadata, metadata_path)
    
    # Вычисляем сколько объектов запрашивать (максимум 10)
    k = min(10, index.ntotal)
    if k == 0:
        return SearchResponse(results=[])
    
    # Поиск в FAISS
    distances, indices = await asyncio.to_thread(index.search, query_vector, k)
    
    search_results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1: 
            continue
        
        meta_item = metadata.get(str(idx), {})
        search_results.append(
            SearchResultItem(
                image_path=meta_item.get("image_path", "unknown"),
                crop_path=meta_item.get("crop_path"),
                center=meta_item.get("center"),
                score=float(dist)
            )
        )
    
    return SearchResponse(results=search_results)


# 4. Отдать содержимое файла index.html
@app.get("/", response_class=FileResponse)
async def read_index():
    index_file_path = "index.html"

    if not os.path.exists(index_file_path):
        raise HTTPException(
            status_code=404, 
            detail="Файл index.html не найден в корневой директории сервиса."
        )

    return FileResponse(index_file_path, media_type="text/html")

def compress_image_jpeg(file_path: str, target_size_bytes: int) -> bytes:
    """
    Уменьшает качество JPEG-изображения без изменения оригинальных размеров (W x H),
    пока размер файла в байтах не станет меньше target_size_bytes.
    """
    with Image.open(file_path) as img:
        img_rgb = img.convert("RGB")
        
        # Перебираем качество с шагом вниз
        for quality in [85, 75, 65, 50, 35, 20]:
            buffer = io.BytesIO()
            img_rgb.save(buffer, format="JPEG", quality=quality, optimize=True)
            data = buffer.getvalue()
            if len(data) <= target_size_bytes:
                return data
                
        # Если даже на низком качестве размер чуть выше лимита, возвращаем последний результат
        return data

@app.get("/api/v1/data/{user_id}/images/{file_path:path}")
async def get_user_image(
    user_id: UUID4, 
    file_path: str, 
    optimize_image: bool = True,
    optimize_image_size: int = 80  # Размер порога оптимизации в килобайтах (по умолчанию 80 КБ)
):
    user_str = str(user_id)
    _, _, images_dir = get_user_paths(user_str)
    
    # Защита от Path Traversal
    base_dir = os.path.abspath(images_dir)
    target_path = os.path.abspath(os.path.join(images_dir, file_path))

    ext = os.path.splitext(target_path)[1].lower()
    media_type = "image/png" if ext == ".png" else "image/jpeg"
    
    if not target_path.startswith(base_dir):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Доступ запрещен."
        )
    
    if not os.path.exists(target_path) or not os.path.isfile(target_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Файл не найден."
        )
    
    file_size = os.path.getsize(target_path)
    target_limit_bytes = optimize_image_size * 1024
    
    # Если оптимизация включена и размер файла превышает заданный лимит
    if optimize_image and file_size > target_limit_bytes:
        compressed_bytes = await asyncio.to_thread(
            compress_image_jpeg, 
            target_path, 
            target_limit_bytes
        )
        return Response(content=compressed_bytes, media_type=media_type)
        
    # В противном случае отдаем файл как есть
    return FileResponse(target_path, media_type=media_type)


class IndexedImageItem(BaseModel):
    image_path: str = Field(..., description="Путь к исходному изображению")
    masks_count: int = Field(..., description="Количество проиндексированных масок для данного изображения")

class UserImagesInfoResponse(BaseModel):
    images: List[IndexedImageItem]


from collections import Counter


@app.get("/api/v1/masks/images/{user_id}", response_model=UserImagesInfoResponse)
async def get_user_images_info(user_id: UUID4):
    user_str = str(user_id)
    _, metadata_path, _ = get_user_paths(user_str)
    
    # Если файла метаданных нет, возвращаем пустой список
    if not os.path.exists(metadata_path):
        return UserImagesInfoResponse(images=[])
    
    # Читаем метаданные асинхронно через поток
    metadata = await asyncio.to_thread(load_metadata, metadata_path)
    
    # Подсчитываем количество масок (записей векторов) для каждого уникального image_path
    image_counts = Counter()
    for item in metadata.values():
        img_path = item.get("image_path")
        if img_path:
            image_counts[img_path] += 1
            
    images_list = [
        IndexedImageItem(image_path=img_path, masks_count=count)
        for img_path, count in image_counts.items()
    ]
    
    return UserImagesInfoResponse(images=images_list)

# --- ЗАПУСК ПРИЛОЖЕНИЯ ---
if __name__ == "__main__":
    import uvicorn
    # Запуск сервера на порту 8000 с поддержкой автоматической перезагрузки (reload)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
