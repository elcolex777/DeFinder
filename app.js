const BASE_URL = 'http://46.16.36.127:8001';

let selectedMaskIndex = null;
let cachedDecodedMasks = [];

// Элементы панели поиска
const searchPanel = document.getElementById("search-panel");
const searchQueryCrop = document.getElementById("search-query-crop");
const searchStatusBadge = document.getElementById("search-status-badge");
const searchFullPreview = document.getElementById("search-full-preview");
const searchFullImg = document.getElementById("search-full-img");
const searchGrid = document.getElementById("search-grid");

let currentSearchAbortController = null;



function hideSearchPanel() {
    if (currentSearchAbortController) {
        currentSearchAbortController.abort();
        currentSearchAbortController = null;
    }
    searchPanel.style.display = "none";
    searchGrid.innerHTML = "";
    searchFullPreview.style.display = "none";
    searchFullImg.removeAttribute("src");
}

/**
 * 1. Получение Bounding Box маски, масштабирование на оригинальное изображение,
 * 2. Ресайз кропа до 200px по большей стороне (с сохранением пропорций),
 * 3. Экспорт кропа в Base64.
 */
function getScaledMaskCropBase64(binaryMask, infW, infH) {
    if (!originalImageElement) return null;
    
    let minX = infW, maxX = -1, minY = infH, maxY = -1;
    
    for (let y = 0; y < infH; y++) {
        const row = y * infW;
        for (let x = 0; x < infW; x++) {
            if (binaryMask[row + x] === 1) {
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
    }
    
    if (maxX < minX || maxY < minY) return null;
    
    const scaleX = originalImageElement.width / infW;
    const scaleY = originalImageElement.height / infH;
    
    // Масштабируем координаты на оригинальное изображение
    const origCropX = Math.max(0, Math.floor(minX * scaleX));
    const origCropY = Math.max(0, Math.floor(minY * scaleY));
    const origCropW = Math.min(originalImageElement.width - origCropX, Math.ceil((maxX - minX + 1) * scaleX));
    const origCropH = Math.min(originalImageElement.height - origCropY, Math.ceil((maxY - minY + 1) * scaleY));
    
    if (origCropW <= 0 || origCropH <= 0) return null;
    
    // Ограничение максимальной стороны до 200px
    let finalW = origCropW;
    let finalH = origCropH;
    const maxDimension = Math.max(origCropW, origCropH);
    
    if (maxDimension > 200) {
        const scaleFactor = 200 / maxDimension;
        finalW = Math.max(1, Math.round(origCropW * scaleFactor));
        finalH = Math.max(1, Math.round(origCropH * scaleFactor));
    }
    
    const cropCanvas = document.createElement("canvas");
    cropCanvas.width = finalW;
    cropCanvas.height = finalH;
    const cropCtx = cropCanvas.getContext("2d");
    
    cropCtx.drawImage(
        originalImageElement,
        origCropX, origCropY, origCropW, origCropH,
        0, 0, finalW, finalH
    );
    
    const dataUrl = cropCanvas.toDataURL("image/jpeg", 0.90);
    return {
        base64WithoutPrefix: dataUrl.split(",")[1],
        fullDataUrl: dataUrl
    };
}

/**
 * 4. Запрос поиска похожих объектов по выбранной маске
 */
async function sendSearchRequest(cropBase64Data) {
    if (currentSearchAbortController) {
        currentSearchAbortController.abort();
    }
    currentSearchAbortController = new AbortController();
    
    searchPanel.style.display = "flex";
    searchQueryCrop.src = cropBase64Data.fullDataUrl;
    searchStatusBadge.className = "queue-badge badge-processing";
    searchStatusBadge.textContent = "Отправка...";
    searchGrid.innerHTML = "";
    searchFullPreview.style.display = "none";
    
    const payload = {
        user_id: currentUserId,
        image_base64: cropBase64Data.base64WithoutPrefix
    };
    
    try {
        const response = await fetch(`${BASE_URL}/api/v1/masks/search`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
            signal: currentSearchAbortController.signal
        });
        
        if (!response.ok) {
            throw new Error(`Ошибка: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        const results = data.results || [];
        
        searchStatusBadge.className = "queue-badge badge-success";
        searchStatusBadge.textContent = `Успешно (${results.length})`;
        
        renderSearchResults(results);
        
    } catch (error) {
        if (error.name === 'AbortError') return;
        searchStatusBadge.className = "queue-badge badge-error";
        searchStatusBadge.textContent = "Ошибка";
        console.error("Search failed:", error);
    }
}

/**
 * Рендеринг карточек результатов в две колонки
 */
function renderSearchResults(results) {
    searchGrid.innerHTML = "";
    
    if (results.length === 0) {
        searchGrid.innerHTML = `<div style="grid-column: span 2; font-size: 13px; color: #64748b; text-align: center; padding: 10px;">Ничего не найдено</div>`;
        return;
    }
    
    results.forEach(item => {
        const card = document.createElement("div");
        card.className = "search-card";
        
        // Форматирование score в 00.00
        const formattedScore = formatScore(item.score);
        
        // Ссылка на кроп маски: /api/v1/{crop_path}
        const cropUrl = `${BASE_URL}/api/v1/${item.crop_path.replace(/^\/+/, '')}`;
        const fullImageUrl = `${BASE_URL}/api/v1/${item.image_path.replace(/^\/+/, '')}`;
        
        card.innerHTML = `
            <img src="${cropUrl}" alt="crop result" loading="lazy">
            <span class="search-score-badge">${formattedScore}</span>
        `;
        
        // При клике на результат показываем полный image_path под шапкой
        card.onclick = () => {
            searchFullImg.src = fullImageUrl;
            searchFullPreview.style.display = "flex";
            searchFullPreview.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        };
        
        searchGrid.appendChild(card);
    });
}

/**
 * Приведение score к формату 00.00 (например, 0.85 -> 00.85, -0.4 -> -00.40)
 */
function formatScore(score) {
    if (typeof score !== "number" || isNaN(score)) return "00.00";
    const sign = score < 0 ? "-" : "";
    const absVal = Math.abs(score);
    const parts = absVal.toFixed(2).split(".");
    const integerPart = parts[0].padStart(2, "0");
    const decimalPart = parts[1];
    return `${sign}${integerPart}.${decimalPart}`;
}




/**
 * Получает user_id из URL или генерирует новый UUID v4
 */
function getOrCreateUserId() {
    const url = new URL(window.location.href);
    let userId = url.searchParams.get("user_id");

    if (!userId) {
        if (typeof crypto !== "undefined" && crypto.randomUUID) {
            userId = crypto.randomUUID();
        } else {
            userId = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
                const r = Math.random() * 16 | 0;
                const v = c === 'x' ? r : (r & 0x3 | 0x8);
                return v.toString(16);
            });
        }
        url.searchParams.set("user_id", userId);
        window.history.replaceState({}, "", url.toString());
    }
    return userId;
}

const currentUserId = getOrCreateUserId();

const cameraInput = document.getElementById("camera-input");
const galleryInput = document.getElementById("gallery-input");
const captureBtn = document.getElementById("capture-btn");
const galleryBtn = document.getElementById("gallery-btn");
const saveBtn = document.getElementById("save-btn");

const resultCanvas = document.getElementById("result-canvas");
const previewZone = document.getElementById("preview-zone");
const settingsPanel = document.getElementById("settings-panel");
const resultsPanel = document.getElementById("results-panel");

const confSlider = document.getElementById("conf-slider");
const iouSlider = document.getElementById("iou-slider");
const confVal = document.getElementById("conf-val");
const iouVal = document.getElementById("iou-val");

const imgszSlider = document.getElementById("imgsz-slider");
const imgszVal = document.getElementById("imgsz-val");
const minwidthSlider = document.getElementById("minwidth-slider");
const minwidthVal = document.getElementById("minwidth-val");

const loader = document.getElementById("loader");
const loaderText = document.getElementById("loader-text");
const errorMessage = document.getElementById("error-message");
const successMessage = document.getElementById("success-message");

const queuePanel = document.getElementById("queue-panel");
const queueList = document.getElementById("queue-list");
const queueCounter = document.getElementById("queue-counter");

let originalImageElement = null;
let lastImgBlobUrl = null;

let lastPredictedMasks = null;
let lastInferenceW = 0;
let lastInferenceH = 0;
let isPredicting = false;

// Очередь задач в памяти
const saveQueue = [];
let isQueueWorkerRunning = false;
let taskIdSequence = 1;

function updateSaveButtonState() {
    const canSave = !isPredicting && 
                    Array.isArray(lastPredictedMasks) && 
                    lastPredictedMasks.length > 0;
    saveBtn.disabled = !canSave;
}

// Сброс панели поиска при сбросе общего состояния
function resetState() {
    if (lastImgBlobUrl) {
        URL.revokeObjectURL(lastImgBlobUrl);
        lastImgBlobUrl = null;
    }
    cameraInput.value = "";
    galleryInput.value = "";
    previewZone.style.display = "none";
    settingsPanel.style.display = "none";
    settingsPanel.open = false;
    resultsPanel.style.display = "none";
    originalImageElement = null;
    lastPredictedMasks = null;
    lastInferenceW = 0;
    lastInferenceH = 0;
    
    selectedMaskIndex = null;
    cachedDecodedMasks = [];
    resultCanvas.style.cursor = "default";
    
    hideSearchPanel();
    updateSaveButtonState();
}

captureBtn.addEventListener("click", function() {
    resetState();
    cameraInput.click();
});

galleryBtn.addEventListener("click", function() {
    resetState();
    galleryInput.click();
});

saveBtn.addEventListener("click", enqueueSaveTask);


/**
 * Обработка клика по холсту
 */
resultCanvas.addEventListener("click", function(event) {
    if (!lastPredictedMasks || cachedDecodedMasks.length === 0 || !lastInferenceW || !lastInferenceH) {
        return;
    }
    
    const rect = resultCanvas.getBoundingClientRect();
    const clickX = (event.clientX - rect.left) * (resultCanvas.width / rect.width);
    const clickY = (event.clientY - rect.top) * (resultCanvas.height / rect.height);
    
    const scaleX = resultCanvas.width / lastInferenceW;
    const scaleY = resultCanvas.height / lastInferenceH;
    
    const infX = Math.floor(clickX / scaleX);
    const infY = Math.floor(clickY / scaleY);
    
    if (infX < 0 || infX >= lastInferenceW || infY < 0 || infY >= lastInferenceH) {
        return;
    }
    
    const pixelPos = infY * lastInferenceW + infX;
    let clickedMaskIndex = -1;
    
    for (let i = cachedDecodedMasks.length - 1; i >= 0; i--) {
        if (cachedDecodedMasks[i].binaryMask[pixelPos] === 1) {
            clickedMaskIndex = i;
            break;
        }
    }
    
    if (clickedMaskIndex !== -1) {
        if (selectedMaskIndex === clickedMaskIndex) {
            // Повторный клик — сброс выбора
            selectedMaskIndex = null;
            hideSearchPanel();
        } else {
            // Выбор маски — запускаем кроп и поиск
            selectedMaskIndex = clickedMaskIndex;
            const maskObj = cachedDecodedMasks[selectedMaskIndex];
            const cropData = getScaledMaskCropBase64(maskObj.binaryMask, lastInferenceW, lastInferenceH);
            if (cropData) {
                sendSearchRequest(cropData);
            }
        }
    } else {
        // Клик в пустое место
        selectedMaskIndex = null;
        hideSearchPanel();
    }
    
    drawMaskBorders(lastPredictedMasks, lastInferenceW, lastInferenceH);
});

function handleFileSelect(event) {
    const files = event.target.files;
    if (files && files.length > 0) {
        const firstFile = files[0];
        lastImgBlobUrl = URL.createObjectURL(firstFile);
        previewZone.style.display = "flex";
        captureBtn.textContent = "📷 Сделать новый снимок";
        galleryBtn.textContent = "🖼️ Выбрать другое фото";
        settingsPanel.style.display = "block";
        resultsPanel.style.display = "flex";
        
        originalImageElement = new Image();
        originalImageElement.onload = function() {
            resultCanvas.width = originalImageElement.width;
            resultCanvas.height = originalImageElement.height;
            const ctx = resultCanvas.getContext("2d");
            ctx.drawImage(originalImageElement, 0, 0);
            sendMasksRequest();
        };
        originalImageElement.src = lastImgBlobUrl;
    }
}

cameraInput.addEventListener("change", handleFileSelect);
galleryInput.addEventListener("change", handleFileSelect);

confSlider.addEventListener("input", () => confVal.textContent = parseFloat(confSlider.value).toFixed(2));
iouSlider.addEventListener("input", () => iouVal.textContent = parseFloat(iouSlider.value).toFixed(2));
imgszSlider.addEventListener("input", () => imgszVal.textContent = parseInt(imgszSlider.value, 10));
minwidthSlider.addEventListener("input", () => minwidthVal.textContent = parseInt(minwidthSlider.value, 10));

confSlider.addEventListener("change", () => { if (originalImageElement) sendMasksRequest(); });
iouSlider.addEventListener("change", () => { if (originalImageElement) sendMasksRequest(); });
imgszSlider.addEventListener("change", () => { if (originalImageElement) sendMasksRequest(); });
minwidthSlider.addEventListener("change", () => { if (originalImageElement) sendMasksRequest(); });

function processAndGetBase64(maxImgSize) {
    if (!originalImageElement) return null;
    let targetWidth = originalImageElement.width;
    let targetHeight = originalImageElement.height;
    
    if (maxImgSize > 0 && targetWidth > maxImgSize) {
        const scaleFactor = maxImgSize / targetWidth;
        targetWidth = maxImgSize;
        targetHeight = Math.round(targetHeight * scaleFactor);
    }
    
    const virtualCanvas = document.createElement("canvas");
    virtualCanvas.width = targetWidth;
    virtualCanvas.height = targetHeight;
    const virtualCtx = virtualCanvas.getContext("2d");
    virtualCtx.drawImage(originalImageElement, 0, 0, targetWidth, targetHeight);
    
    const dataUrl = virtualCanvas.toDataURL("image/jpeg", 0.90);
    return dataUrl.split(",")[1];
}

function decodeRLE(rleMask, width, height) {
    const binaryMask = new Uint8Array(width * height);
    let pixelIndex = 0;
    for (let i = 0; i < rleMask.length; i++) {
        const runLength = rleMask[i];
        const pixelValue = (i % 2 === 1) ? 1 : 0;
        for (let j = 0; j < runLength; j++) {
            binaryMask[pixelIndex++] = pixelValue;
        }
    }
    return binaryMask;
}

function scaleAndEncodeRleMask(rleMask, infW, infH, origW, origH) {
    const infBinaryMask = decodeRLE(rleMask, infW, infH);
    const scaleX = origW / infW;
    const scaleY = origH / infH;
    
    const encodedRle = [];
    let expectedVal = 0;
    let currentRun = 0;
    
    for (let yOrig = 0; yOrig < origH; yOrig++) {
        const yInf = Math.min(Math.floor(yOrig / scaleY), infH - 1);
        const rowOffsetInf = yInf * infW;
        
        for (let xOrig = 0; xOrig < origW; xOrig++) {
            const xInf = Math.min(Math.floor(xOrig / scaleX), infW - 1);
            const val = infBinaryMask[rowOffsetInf + xInf];
            
            if (val === expectedVal) {
                currentRun++;
            } else {
                encodedRle.push(currentRun);
                expectedVal = 1 - expectedVal;
                currentRun = 1;
            }
        }
    }
    encodedRle.push(currentRun);
    return encodedRle;
}

function drawMaskBorders(masksArray, inferenceWidth, inferenceHeight) {
    const ctx = resultCanvas.getContext("2d");
    ctx.clearRect(0, 0, resultCanvas.width, resultCanvas.height);
    ctx.drawImage(originalImageElement, 0, 0, resultCanvas.width, resultCanvas.height);
    
    const scaleX = resultCanvas.width / inferenceWidth;
    const scaleY = resultCanvas.height / inferenceHeight;
    
    // Подготовка кэша распакованных масок и постоянных оттенков
    if (cachedDecodedMasks.length !== masksArray.length) {
        cachedDecodedMasks = masksArray.map((rleMask, idx) => ({
            binaryMask: decodeRLE(rleMask, inferenceWidth, inferenceHeight),
            hue: Math.floor((idx * 137.5) % 360) // Равномерное и детерминированное распределение цветов
        }));
    }
    
    cachedDecodedMasks.forEach((item, index) => {
        // Если выбрана конкретная маска, остальные скрываем
        if (selectedMaskIndex !== null && selectedMaskIndex !== index) {
            return;
        }
        
        const binaryMask = item.binaryMask;
        const isSelected = (selectedMaskIndex === index);
        
        // Для выбранной маски делаем фон чуть ярче и линию толще
        const fillAlpha = isSelected ? 0.55 : 0.35;
        const strokeWidth = isSelected ? 4 : 3;
        
        const fillColor = `hsla(${item.hue}, 100%, 50%, ${fillAlpha})`;
        const strokeColor = `hsl(${item.hue}, 100%, 50%)`;
        
        const fillPath = new Path2D();
        const borderPath = new Path2D();
        
        for (let y = 0; y < inferenceHeight; y++) {
            for (let x = 0; x < inferenceWidth; x++) {
                const currentPos = y * inferenceWidth + x;
                
                if (binaryMask[currentPos] === 1) {
                    const rectX = x * scaleX;
                    const rectY = y * scaleY;
                    
                    fillPath.rect(rectX, rectY, scaleX, scaleY);
                    
                    const isLeftBorder   = (x <= 0) || (binaryMask[currentPos - 1] !== 1);
                    const isRightBorder  = (x >= inferenceWidth - 1) || (binaryMask[currentPos + 1] !== 1);
                    const isTopBorder    = (y <= 0) || (binaryMask[(y - 1) * inferenceWidth + x] !== 1);
                    const isBottomBorder = (y >= inferenceHeight - 1) || (binaryMask[(y + 1) * inferenceWidth + x] !== 1);
                    
                    if (isLeftBorder || isRightBorder || isTopBorder || isBottomBorder) {
                        borderPath.rect(rectX, rectY, scaleX, scaleY);
                    }
                }
            }
        }
        
        ctx.fillStyle = fillColor;
        ctx.fill(fillPath);
        
        ctx.lineWidth = strokeWidth;
        ctx.strokeStyle = strokeColor;
        ctx.stroke(borderPath);
    });
}

async function sendMasksRequest() {
    if (!originalImageElement) return;
    
    const currentImgsz = parseInt(imgszSlider.value, 10);
    const optimizedBase64 = processAndGetBase64(currentImgsz);
    if (!optimizedBase64) return;
    
    isPredicting = true;
    lastPredictedMasks = null;
    selectedMaskIndex = null;
    cachedDecodedMasks = [];
    resultCanvas.style.cursor = "default";
    updateSaveButtonState();
    
    loaderText.textContent = "Обработка масок...";
    loader.style.display = "flex";
    errorMessage.style.display = "none";
    successMessage.style.display = "none";
    
    const requestPayload = {
        image_base64: optimizedBase64,
        conf_threshold: parseFloat(confSlider.value),
        iou_threshold: parseFloat(iouSlider.value),
        imgsz: currentImgsz,
        min_width: parseFloat(minwidthSlider.value)
    };
    
    try {
        const response = await fetch(`${BASE_URL}/api/v1/masks/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(requestPayload)
        });
        
        if (!response.ok) {
            throw new Error(`Ошибка сервера: ${response.status} ${response.statusText}`);
        }
        
        const responseData = await response.json();
        if (!responseData || !Array.isArray(responseData.masks)) {
            throw new Error("Ответ сервера не содержит массив 'masks'");
        }
        
        let infW = originalImageElement.width;
        let infH = originalImageElement.height;
        if (currentImgsz > 0 && infW > currentImgsz) {
            const ratio = currentImgsz / infW;
            infW = currentImgsz;
            infH = Math.round(infH * ratio);
        }
        
        lastInferenceW = infW;
        lastInferenceH = infH;
        lastPredictedMasks = responseData.masks;
        cachedDecodedMasks = []; // обнуляем кэш под новые маски
        selectedMaskIndex = null;
        resultCanvas.style.cursor = "pointer";
        
        const totalMasksCount = responseData.masks.length;
        successMessage.textContent = `Успешно получено масок: ${totalMasksCount}`;
        successMessage.style.display = "block";
        
        drawMaskBorders(responseData.masks, lastInferenceW, lastInferenceH);
        
    } catch (error) {
        errorMessage.textContent = `Не удалось получить маски: ${error.message}`;
        errorMessage.style.display = "block";
    } finally {
        isPredicting = false;
        loader.style.display = "none";
        updateSaveButtonState();
    }
}

// -------------------------------------------------------------
// РАБОТА С ОЧЕРЕДЬЮ СОХРАНЕНИЯ В ПАМЯТИ
// -------------------------------------------------------------

function renderQueueUI() {
    if (saveQueue.length === 0) {
        queuePanel.style.display = "none";
        return;
    }
    
    queuePanel.style.display = "flex";
    queueCounter.textContent = `${saveQueue.length} ${getNoun(saveQueue.length, 'задача', 'задачи', 'задач')}`;
    queueList.innerHTML = "";
    
    saveQueue.forEach(task => {
        const item = document.createElement("div");
        item.className = "queue-item";
        
        let badgeClass = "badge-pending";
        let statusText = "В очереди";
        
        if (task.status === "processing") {
            badgeClass = "badge-processing";
            statusText = "Обрабатывается...";
        } else if (task.status === "success") {
            badgeClass = "badge-success";
            statusText = "Успешно сохранено";
        } else if (task.status === "error") {
            badgeClass = "badge-error";
            statusText = "Ошибка";
        }
        
        item.innerHTML = `
            <div class="queue-item-header">
                <span class="queue-item-title">Задача #${task.id} (${task.masksCount} масок)</span>
                <span class="queue-badge ${badgeClass}">${statusText}</span>
            </div>
            ${task.message ? `<div class="queue-item-msg">${task.message}</div>` : ''}
        `;
        
        if (task.status === "error") {
            const retryBtn = document.createElement("button");
            retryBtn.className = "btn-retry";
            retryBtn.textContent = "🔄 Повторить";
            retryBtn.onclick = () => retrySaveTask(task.id);
            item.appendChild(retryBtn);
        }
        
        queueList.appendChild(item);
    });
}

function getNoun(number, one, two, five) {
    let n = Math.abs(number);
    n %= 100;
    if (n >= 5 && n <= 20) return five;
    n %= 10;
    if (n === 1) return one;
    if (n >= 2 && n <= 4) return two;
    return five;
}

/**
 * Добавление новой задачи в очередь
 */
function enqueueSaveTask() {
    if (!originalImageElement || !lastPredictedMasks || lastPredictedMasks.length === 0) return;
    
    const originalBase64 = processAndGetBase64(0);
    if (!originalBase64) return;
    
    const origW = originalImageElement.width;
    const origH = originalImageElement.height;
    
    // Снапшот смасштабированных масок для изолированной отправки
    const scaledMasks = lastPredictedMasks.map(mask => 
        scaleAndEncodeRleMask(mask, lastInferenceW, lastInferenceH, origW, origH)
    );
    
    const task = {
        id: taskIdSequence++,
        createdAt: new Date(),
        status: "pending", // pending | processing | success | error
        message: "",
        masksCount: scaledMasks.length,
        payload: {
            user_id: currentUserId,
            image_base64: originalBase64,
            masks: scaledMasks
        }
    };
    
    saveQueue.push(task);
    renderQueueUI();
    
    processQueueWorker();
}

/**
 * Повторная попытка для задачи со статусом "error"
 */
function retrySaveTask(taskId) {
    const task = saveQueue.find(t => t.id === taskId);
    if (task && task.status === "error") {
        task.status = "pending";
        task.message = "Ожидание повторной отправки...";
        renderQueueUI();
        processQueueWorker();
    }
}

/**
 * Последовательный воркер обработки очереди
 */
async function processQueueWorker() {
    if (isQueueWorkerRunning) return;
    
    // Ищем первую задачу, готовую к обработке
    const nextTask = saveQueue.find(task => task.status === "pending");
    if (!nextTask) return;
    
    isQueueWorkerRunning = true;
    nextTask.status = "processing";
    nextTask.message = "Отправка запроса на сохранение...";
    renderQueueUI();
    
    try {
        const response = await fetch(`${BASE_URL}/api/v1/masks/index`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(nextTask.payload)
        });
        
        if (!response.ok) {
            throw new Error(`Ошибка сервера: ${response.status} ${response.statusText}`);
        }
        
        const responseData = await response.json();
        nextTask.status = "success";
        nextTask.message = responseData.message || "Индексация успешно завершена.";
        renderQueueUI();
        
        // Исчезает через 5 секунд после успешного сохранения
        setTimeout(() => {
            const index = saveQueue.findIndex(t => t.id === nextTask.id);
            if (index !== -1) {
                saveQueue.splice(index, 1);
                renderQueueUI();
            }
        }, 5000);
        
    } catch (error) {
        nextTask.status = "error";
        nextTask.message = error.message || "Неизвестная ошибка";
        renderQueueUI();
    } finally {
        isQueueWorkerRunning = false;
        // Запуск следующей задачи в очереди
        processQueueWorker();
    }
}