const BASE_URL = 'http://46.16.36.127:8001';

const cameraInput = document.getElementById("camera-input");
const captureBtn = document.getElementById("capture-btn");
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
const errorMessage = document.getElementById("error-message");
const successMessage = document.getElementById("success-message");

let originalImageElement = null;
let lastImgBlobUrl = null;

captureBtn.addEventListener("click", function() {
    if (lastImgBlobUrl) {
        URL.revokeObjectURL(lastImgBlobUrl);
        lastImgBlobUrl = null;
    }
    cameraInput.value = "";
    previewZone.style.display = "none";
    settingsPanel.style.display = "none";
    resultsPanel.style.display = "none";
    originalImageElement = null;
    cameraInput.click();
});

cameraInput.addEventListener("change", function(event) {
    const files = event.target.files;
    if (files && files.length > 0) {
        const firstFile = files[0];
        lastImgBlobUrl = URL.createObjectURL(firstFile);
        previewZone.style.display = "flex";
        captureBtn.textContent = "Сделать новый снимок";
        settingsPanel.style.display = "flex";
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
});

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
    
    const dataUrl = virtualCanvas.toDataURL("image/jpeg", 0.85);
    return dataUrl.split(",")[1];
}

function drawMaskBorders(masksArray, inferenceWidth, inferenceHeight) {
    const ctx = resultCanvas.getContext("2d");
    ctx.clearRect(0, 0, resultCanvas.width, resultCanvas.height);
    ctx.drawImage(originalImageElement, 0, 0, resultCanvas.width, resultCanvas.height);
    
    const scaleX = resultCanvas.width / inferenceWidth;
    const scaleY = resultCanvas.height / inferenceHeight;
    
    masksArray.forEach(rleMask => {
        let pixelIndex = 0;
        const binaryMask = new Uint8Array(inferenceWidth * inferenceHeight);
        
        for (let i = 0; i < rleMask.length; i++) {
            const runLength = rleMask[i];
            const pixelValue = (i % 2 === 1) ? 1 : 0;
            for (let j = 0; j < runLength; j++) {
                binaryMask[pixelIndex++] = pixelValue;
            }
        }
        
        ctx.lineWidth = 3;
        const randomHue = Math.floor(Math.random() * 360);
        ctx.strokeStyle = `hsl(${randomHue}, 100%, 50%)`;
        ctx.beginPath();
        
        for (let y = 0; y < inferenceHeight; y++) {
            for (let x = 0; x < inferenceWidth; x++) {
                const currentPos = y * inferenceWidth + x;
                if (binaryMask[currentPos] === 1) {
                    let isLeftBorder = (x <= 0) || (binaryMask[currentPos - 1] !== 1);
                    let isRightBorder = (x >= inferenceWidth - 1) || (binaryMask[currentPos + 1] !== 1);
                    let isTopBorder = (y <= 0) || (binaryMask[(y - 1) * inferenceWidth + x] !== 1);
                    let isBottomBorder = (y >= inferenceHeight - 1) || (binaryMask[(y + 1) * inferenceWidth + x] !== 1);
                    
                    if (isLeftBorder || isRightBorder || isTopBorder || isBottomBorder) {
                        ctx.rect(x * scaleX, y * scaleY, scaleX, scaleY);
                    }
                }
            }
        }
        ctx.stroke();
    });
}

async function sendMasksRequest() {
    if (!originalImageElement) return;
    
    const currentImgsz = parseInt(imgszSlider.value, 10);
    const optimizedBase64 = processAndGetBase64(currentImgsz);
    if (!optimizedBase64) return;
    
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
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(requestPayload)
        });
        
        if (!response.ok) {
            throw new Error(`Ошибка сервера: ${response.status} ${response.statusText}`);
        }
        
        const responseData = await response.json();
        if (!responseData || !Array.isArray(responseData.masks)) {
            throw new Error("Ответ сервера не содержит массив 'masks'");
        }
        
        const totalMasksCount = responseData.masks.length;
        successMessage.textContent = `Успешно получено масок: ${totalMasksCount}`;
        successMessage.style.display = "block";
        
        let inferenceW = originalImageElement.width;
        let inferenceH = originalImageElement.height;
        if (currentImgsz > 0 && inferenceW > currentImgsz) {
            const ratio = currentImgsz / inferenceW;
            inferenceW = currentImgsz;
            inferenceH = Math.round(inferenceH * ratio);
        }
        
        drawMaskBorders(responseData.masks, inferenceW, inferenceH);
        
    } catch (error) {
        errorMessage.textContent = `Не удалось получить маски: ${error.message}`;
        errorMessage.style.display = "block";
    } finally {
        loader.style.display = "none";
    }
}
