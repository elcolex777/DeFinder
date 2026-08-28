# DeFinder
Find thing in image

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

```

## Конфигурация nginx
```
echo 'server {
    listen 8001;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Увеличиваем лимит на размер тела запроса, так как base64 изображения могут быть большими
        client_max_body_size 50M;
    }
}' | sudo tee /etc/nginx/sites-available/definder > /dev/null && \
sudo ln -sf /etc/nginx/sites-available/definder /etc/nginx/sites-enabled/ && \
sudo nginx -t && \
sudo systemctl restart nginx

```
