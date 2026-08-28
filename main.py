import os
import io
import json
import uuid
import base64
from typing import List
from pydantic import BaseModel, Field, UUID4
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

import torch
import numpy as np
import faiss
from PIL import Image

# Импорты ваших моделей
from ultralytics import SAM
import open_clip

app = FastAPI(title="MobileSAM & CLIP FAISS Indexing Service")

# --- ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ ---
# Рекомендуется использовать 'cuda', если есть GPU, иначе 'cpu'
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading MobileSAM model on {device}...")
sam_model = SAM("mobile_sam.pt")

print(f"Loading CLIP model (ViT-B-32) on {device}...")
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion20c_e32')
clip_model = clip_model.to(device)
clip_model.eval()

# Размерность вектора для ViT-B-32 равна 512
DIMENSION = 512


# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
def decode_base64_image(base64_str: str) -> Image.Image:
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
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


# --- PYDANTIC СХЕМЫ ДАННЫХ ---
class MaskPredictionRequest(BaseModel):
    image_base64: str = Field(..., description="Изображение в формате Base64")
    conf_threshold: float = Field(0.25, description="Порог уверенности (conf) для MobileSAM")
    iou_threshold: float = Field(0.7, description="Порог IoU для NMS в MobileSAM")

class MaskPredictionResponse(BaseModel):
    masks: List[List[List[int]]] = Field(..., description="Массив бинарных масок [N, H, W]")

class IndexMasksRequest(BaseModel):
    user_id: UUID4 = Field(..., description="GUID пользователя")
    image_base64: str = Field(..., description="Изображение в формате Base64")
    masks: List[List[List[int]]] = Field(..., description="Массив масок из первого эндпоинта")

class SearchRequest(BaseModel):
    user_id: UUID4 = Field(..., description="GUID пользователя")
    image_base64: str = Field(..., description="Изображение для поиска в Base64")

class SearchResultItem(BaseModel):
    image_path: str
    score: float = Field(..., description="Косинусное сходство (от -1 до 1, чем выше — тем ближе)")

class SearchResponse(BaseModel):
    results: List[SearchResultItem]


# --- КОНЕЧНЫЕ ТОЧКИ (Endpoints) ---

# 1. Получить маски по содержимому изображения
@app.post("/api/v1/masks/predict", response_model=MaskPredictionResponse)
async def predict_masks(payload: MaskPredictionRequest):
    image = decode_base64_image(payload.image_base64)
    
    # Запускаем инференс MobileSAM (по умолчанию принимает PIL Image или пути)
    # Передаем параметры conf и iou напрямую в предиктор ultralytics
    results = sam_model.predict(image, conf=payload.conf_threshold, iou=payload.iou_threshold, verbose=False)
    
    output_masks = []
    if results and results[0].masks is not None:
        # Получаем маски в виде тензора/массива [N, H, W] с типом float/bool
        # Переводим в int (0 и 1) и конвертируем в нативный список Python
        masks_data = results[0].masks.data.cpu().numpy().astype(int)
        output_masks = masks_data.tolist()
        
    return MaskPredictionResponse(masks=output_masks)


# 2. Сохранить маски и проиндексировать
@app.post("/api/v1/masks/index")
async def index_masks(payload: IndexMasksRequest):
    user_str = str(payload.user_id)
    image = decode_base64_image(payload.image_base64)
    
    index_path, metadata_path, images_dir = get_user_paths(user_str)
    
    # Сохраняем изображение на диск: data/{id}/images/{guid}.jpg
    img_filename = f"{uuid.uuid4()}.jpg"
    full_image_path = os.path.join(images_dir, img_filename)
    image.save(full_image_path, format="JPEG")
    
    # Извлекаем и нормализуем эмбеддинг через CLIP ViT-B-32
    embedding = extract_clip_embedding(image)
    
    # Работа с FAISS
    index = load_or_create_index(index_path)
    vector_to_add = np.array([embedding]).astype('float32')
    
    # Порядковый ID вектора в текущем индексе
    current_vector_id = index.ntotal
    index.add(vector_to_add)
    
    # Перезаписываем обновленный индекс на диск
    faiss.write_index(index, index_path)
    
    # Сохраняем связь ID вектора и метаданных на диск
    metadata = load_metadata(metadata_path)
    metadata[str(current_vector_id)] = {
        "image_path": full_image_path,
        "masks_count": len(payload.masks)
    }
    save_metadata(metadata_path, metadata)
    
    return {"status": "ok", "message": "Изображение успешно добавлено в индекс пользователя."}


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
    query_vector = extract_clip_embedding(query_image)
    query_vector = np.array([query_vector]).astype('float32')
    
    # Читаем индекс и метаданные пользователя с диска
    index = faiss.read_index(index_path)
    metadata = load_metadata(metadata_path)
    
    # Вычисляем сколько объектов запрашивать (максимум 10)
    k = min(10, index.ntotal)
    if k == 0:
        return SearchResponse(results=[])
    
    # Поиск в FAISS. Для IndexFlatIP:
    # distances — это значения косинусного сходства (чем БОЛЬШЕ значение, тем ближе картинки)
    distances, indices = index.search(query_vector, k)
    
    search_results = []
    # Извлекаем результаты (массивы двумерные, берем строку 0)
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1: 
            continue
        
        meta_item = metadata.get(str(idx), {})
        search_results.append(
            SearchResultItem(
                image_path=meta_item.get("image_path", "unknown"),
                score=float(dist)
            )
        )
    
    # IndexFlatIP возвращает результаты уже отсортированными по убыванию сходства (от лучших к худшим)
    return SearchResponse(results=search_results)
    
# 4. Отдать содержимое файла index.html
@app.get("/", response_class=FileResponse)
async def read_index():
    # Путь к файлу index.html в корне проекта
    index_file_path = "index.html"

    # Проверяем, существует ли файл, чтобы избежать внутренней ошибки сервера
    if not os.path.exists(index_file_path):
        raise HTTPException(
            status_code=404, 
            detail="Файл index.html не найден в корневой директории сервиса."
        )

    return FileResponse(index_file_path, media_type="text/html")


if __name__ == "__main__":
    import uvicorn
    # Запуск сервера на порту 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
