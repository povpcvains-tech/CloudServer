// face-recognition.js
// Модуль для сравнения лиц с использованием face-api.js

class FaceRecognizer {
    constructor() {
        this.isLoaded = false;
        this.faceDatabase = [];      // Массив base64 изображений
        this.faceDescriptors = [];   // Массив дескрипторов лиц
        this.loadProgress = 0;
    }

    // Загрузка моделей face-api
    async loadModels() {
        const MODEL_URL = 'https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/models';
        
        console.log('Загрузка моделей face-api...');
        await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
        console.log('✓ TinyFaceDetector загружен');
        await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
        console.log('✓ FaceLandmark68 загружен');
        await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
        console.log('✓ FaceRecognitionNet загружен');
        
        this.isLoaded = true;
        return true;
    }

    // Загрузка базы лиц из JSON (base64)
    async loadFaceDatabase(jsonUrl) {
        try {
            const response = await fetch(jsonUrl);
            const data = await response.json();
            this.faceDatabase = data.faces || [];
            console.log(`Загружено ${this.faceDatabase.length} лиц из JSON`);
            return this.faceDatabase.length;
        } catch (err) {
            console.error('Ошибка загрузки базы лиц:', err);
            return 0;
        }
    }

    // Вычисление дескрипторов для всех лиц в базе
    async computeDescriptors() {
        if (!this.isLoaded) {
            throw new Error('Модели не загружены');
        }
        
        this.faceDescriptors = [];
        
        for (let i = 0; i < this.faceDatabase.length; i++) {
            try {
                const img = new Image();
                img.src = this.faceDatabase[i];
                await new Promise((resolve, reject) => {
                    img.onload = resolve;
                    img.onerror = reject;
                });
                
                const detection = await faceapi.detectSingleFace(img)
                    .withFaceLandmarks()
                    .withFaceDescriptor();
                
                if (detection) {
                    this.faceDescriptors.push({
                        descriptor: detection.descriptor,
                        image: this.faceDatabase[i],
                        index: i
                    });
                    console.log(`✓ Лицо ${i + 1} обработано`);
                } else {
                    console.warn(`⚠ Лицо ${i + 1} не обнаружено на фото`);
                }
            } catch (err) {
                console.warn(`Ошибка обработки лица ${i}:`, err);
            }
        }
        
        console.log(`Готово ${this.faceDescriptors.length} дескрипторов лиц`);
        return this.faceDescriptors.length;
    }

    // Сравнение дескриптора с базой
    matchFace(descriptor, threshold = 0.55) {
        if (this.faceDescriptors.length === 0) return null;
        
        let bestMatch = null;
        let bestDistance = threshold;
        
        for (const item of this.faceDescriptors) {
            const distance = faceapi.euclideanDistance(descriptor, item.descriptor);
            if (distance < bestDistance) {
                bestDistance = distance;
                bestMatch = item;
            }
        }
        
        if (bestMatch) {
            const similarity = Math.round((1 - bestDistance) * 100);
            return {
                match: bestMatch,
                distance: bestDistance,
                similarity: similarity
            };
        }
        return null;
    }

    // Распознавание лиц на видео
    async detectFaces(videoElement) {
        if (!this.isLoaded || !videoElement.videoWidth) return [];
        
        try {
            const detections = await faceapi.detectAllFaces(
                videoElement,
                new faceapi.TinyFaceDetectorOptions()
            )
            .withFaceLandmarks()
            .withFaceDescriptors();
            
            return detections;
        } catch (err) {
            console.warn('Ошибка детекции лиц:', err);
            return [];
        }
    }

    // Получить количество загруженных лиц
    getDatabaseCount() {
        return this.faceDescriptors.length;
    }
}

// Экспорт для использования в HTML
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FaceRecognizer;
}
