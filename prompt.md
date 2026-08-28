на основе кода ниже сделать пример fastapi сервиса со следующими конечными точками:

1. получить маски по содержимому изображения. На входе изображение в base64 и параметры для mobilesam (с установленными значениями по умолчанию). На выходе массив масок.

2. сохранить маски и проиндексировать. На входе массив масок, изображение в base64 и id пользователя (строка guid). для каждого пользователя (id) создается отдельный индекс faiss на диске. путь на диске data/{id пользователя}/index/faiss.index
изображение сохраняется на диск в подпапку data/{id пользователя}/images/guid.{ext}
путь к изображению также сохраняется в методанных.
На выходе результат ок.

3. поиск по фото. На входе изображение в base64, id пользователя.
На основе входного изображения делаем вектор и ищем в индексе faiss. 
на выходе возвращает массив из 10 найденных объектов вместе с image path и score




pip install ultralytics torch torchvision open_clip_torch faiss-cpu pillow numpy



import torch
import numpy as np
from PIL import Image
import faiss
from ultralytics import SAM
import open_clip

# ==========================================
# 1. ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ
# ==========================================
# Загружаем MobileSAM через Ultralytics
sam_model = SAM("mobile_sam.pt") 

# Загружаем CLIP для извлечения качественных векторов (эмбеддингов)
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion20c_e32')
clip_model.eval()

# Создаем индекс FAISS для поиска по вектору (размерность 512 для ViT-B-32)
dimension = 512
index = faiss.IndexFlatIP(dimension)  # Индекс для Inner Product (Косинусное сходство при нормализации)

# Списки для хранения метаданных
saved_vectors = []
image_metadata = []

# ==========================================
# 2. ФУНКЦИЯ СЕГМЕНТАЦИИ И ВЕКТОРИЗАЦИИ
# ==========================================
def process_and_index_image(image_path, object_id, prompt_points=None):
    """
    Сегментирует объект, вырезает его, превращает в вектор и добавляет в базу.
    prompt_points: список координат [[x, y]] для выбора объекта. Если None — берется авто-маска.
    """
    img = Image.open(image_path).convert("RGB")
    
    # Шаг 1: Сегментация с помощью MobileSAM
    if prompt_points:
        results = sam_model(image_path, points=prompt_points, labels=[1])
    else:
        results = sam_model(image_path) # Автоматический режим (сегментирует всё)
        
    if not results or len(results[0].masks) == 0:
        print(f"Объекты на {image_path} не найдены.")
        return

    # Берем первую найденную маску
    mask = results[0].masks.data[0].cpu().numpy()
    
    # Шаг 2: Изоляция объекта (наложение маски на изображение)
    img_np = np.array(img)
    # Масштабируем маску под размер оригинального фото, если они различаются
    mask_resized = np.array(Image.fromarray(mask).resize(img.size, resample=Image.NEAREST))
    
    # Создаем изображение, где всё кроме маски — черное
    isolated_object = np.zeros_like(img_np)
    isolated_object[mask_resized == 1] = img_np[mask_resized == 1]
    isolated_img = Image.fromarray(isolated_object)

    # Шаг 3: Получение вектора через CLIP
    image_input = preprocess(isolated_img).unsqueeze(0)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        # Нормализуем вектор (обязательно для косинусного сходства в FAISS)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        vector = image_features.cpu().numpy().flatten()

    # Шаг 4: Добавление в FAISS
    index.add(np.array([vector]).astype('float32'))
    image_metadata.append({"object_id": object_id, "path": image_path})
    print(f"Успешно проиндексирован объект из {image_path}")

# ==========================================
# 3. ФУНКЦИЯ ПОИСКА ПОХОЖЕГО ОБЪЕКТА
# ==========================================
def search_similar_mask(query_vector, top_k=3):
    """Ищет top_k похожих векторов в базе FAISS"""
    query_vector = np.array([query_vector]).astype('float32')
    distances, indices = index.search(query_vector, top_k)
    
    print("\n--- Результаты поиска ---")
    for i, idx in enumerate(indices[0]):
        if idx == -1: continue
        score = distances[0][i]
        meta = image_metadata[idx]
        print(f"Место {i+1}: ID={meta['object_id']}, Файл={meta['path']}, Сходство={score:.4f}")

# ==========================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ==========================================
# 1. Заполняем базу вашими фото (пройдите циклом по папке)
# В prompt_points передаем координаты центра нужного объекта [X, Y]
process_and_index_image("photo_1.jpg", object_id="apple_red", prompt_points=[[250, 300]])
process_and_index_image("photo_2.jpg", object_id="banana_yellow", prompt_points=[[100, 150]])

# 2. Допустим, у нас есть новое фото (запрос), из которого мы извлекли вектор query_vector
# (Для теста вызовем векторизацию отдельно или возьмем вектор существующего объекта)
# Сканируем тестовое изображение, чтобы получить вектор для поиска:
test_img = Image.open("query_photo.jpg").convert("RGB")
test_input = preprocess(test_img).unsqueeze(0)
with torch.no_grad():
    query_feat = clip_model.encode_image(test_input)
    query_feat /= query_feat.norm(dim=-1, keepdim=True)
    query_vector = query_feat.cpu().numpy().flatten()

# 3. Ищем похожие маски в нашей базе
search_similar_mask(query_vector, top_k=2)
