# DeFinder
Поиск вещей по картинке

## Использование

Открыть на смартфоне (или на ПК) в браузере страницу http://46.16.36.127:8001/

1. Сделать снимок или выбрать из галереи.
2. Сохранить это фото с проиндексированными сегментами для дальнейшего поиска.
3. Найти фото на основе снимка. 

При снимке на камеру или выбор из галереи, сразу запускается поиск по всем проиндексированным ранее картинкам.
Можно выбрать конкретный объект мышью (или тапом), поиск запускается сразу. Результаты показываются ниже списком (это части изображения, похожие на выбранный объект) цифра в нижнем углу показывает скор похожести (чем больше тем лучше). На картинку можно кликнуть, показывается оригинал проиндексированного изображения, на котором отображается зеленая квадратная рамка, где этот объект на изображении.

![example](https://raw.githubusercontent.com/elcolex777/DeFinder/refs/heads/main/example.jpg)

Можно выбрать несколько фото из галереи и они отправятся в очередь на индексацию

## Сборка Docker-образа
```
mkdir -p /app && cd /app

curl -L https://github.com/elcolex777/DeFinder/archive/refs/heads/main.zip -o DeFinder.zip
unzip DeFinder.zip

cd /app/DeFinder-main
docker build -t definder-service .

```
## Запуск Docker-контейнера
```
mkdir -p /app/DeFinder-main/data

docker run -d \
  -p 8000:8000 \
  -v /app/DeFinder-main/data:/app/data \
  --name definder_service \
  definder-service

docker logs -n 100 definder_service
docker container rm definder_service
docker cp main.py definder_service:/app/main.py

docker update --restart unless-stopped definder_service

```

## Конфигурация nginx
```
echo 'server {
    listen 8001;
    server_name _;

    location / {

    # Разрешить запросы с любого домена (для разработки)
    # Для продакшена лучше указать конкретный домен, например: https://example.com
    add_header 'Access-Control-Allow-Origin' '*' always;
    
    # Разрешенные методы и заголовки
    add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS, PUT, DELETE' always;
    add_header 'Access-Control-Allow-Headers' 'X-Requested-With,Accept,Content-Type,Origin,Authorization' always;

    # Обработка предварительного (preflight) запроса OPTIONS
    if ($request_method = 'OPTIONS') {
        add_header 'Access-Control-Allow-Origin' '*' always;
        add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS, PUT, DELETE' always;
        add_header 'Access-Control-Allow-Headers' 'X-Requested-With,Accept,Content-Type,Origin,Authorization' always;
        add_header 'Access-Control-Max-Age' 1728000;
        add_header 'Content-Type' 'text/plain; charset=utf-8';
        add_header 'Content-Length' 0;
        return 204;
    }

        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Увеличиваем лимит на размер тела запроса, так как base64 изображения могут быть большими
        client_max_body_size 50M;
        proxy_read_timeout 300s;
    }
}' | sudo tee /etc/nginx/sites-available/definder > /dev/null && \
sudo ln -sf /etc/nginx/sites-available/definder /etc/nginx/sites-enabled/ && \
sudo nginx -t && \
sudo systemctl restart nginx




curl -X POST "http://127.0.0.1:8001/api/v1/masks/predict" \
curl -X POST "http://46.16.36.127:8001/api/v1/masks/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "image_base64": "/9j/4AAQSkZJRg",
       "conf_threshold": 0.25,
       "iou_threshold": 0.7
     }'

```
