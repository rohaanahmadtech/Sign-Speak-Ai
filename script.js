import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";

// Canvas elements
const video = document.getElementById("webcam");
const liveCanvas = document.getElementById("output_canvas");
const liveCtx = liveCanvas.getContext("2d");
const imageCanvas = document.getElementById("image_canvas");

// State
let handLandmarker = null;  // Initialize as null
let model = null;
let currentPrediction = "";
let lastVideoTime = -1;
let activeMode = null;
let cameraStream = null;
let inferenceRunning = false;
let isModelLoaded = false;
let isMediaPipeReady = false;

// Expose to UI
window.currentPrediction = "";
window.clearCurrentPrediction = () => {
    currentPrediction = "";
    window.currentPrediction = "";
    if (window.updatePrediction) window.updatePrediction("", 0);
};

// Label map (must match training)
const labelMap = ["A","B","Blank","C","D","E","F","G","H","I",
    "J","K","L","M","N","O","P","Q","R","S",
    "T","U","V","W","X","Y","Z"];

// Initialize MediaPipe and Model
async function initialize() {
    try {
        console.log("Loading MediaPipe...");
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            },
            runningMode: "VIDEO",
            numHands: 1,
            minHandDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });
        
        isMediaPipeReady = true;
        console.log("✅ MediaPipe HandLandmarker ready");
        
        console.log("Loading TensorFlow model...");
        try {
            model = await tf.loadLayersModel('./web_model/model.json');
            isModelLoaded = true;
            console.log("✅ TensorFlow model loaded successfully");
            console.log("Model input shape:", model.inputs[0].shape);
        } catch (modelError) {
            console.error("Model load error:", modelError);
            isModelLoaded = false;
        }
        
        console.log("✅ AI Engine Ready");
        
        // If live mode is active, start camera
        if (activeMode === 'live') {
            await startCamera();
        }
        
    } catch (error) {
        console.error("Initialization failed:", error);
        isMediaPipeReady = false;
    }
}

// Process landmarks
function processLandmarks(landmarks) {
    const wrist = landmarks[0];
    let coords = [];
    for (let i = 0; i < landmarks.length; i++) {
        coords.push(landmarks[i].x - wrist.x);
        coords.push(landmarks[i].y - wrist.y);
        coords.push(landmarks[i].z - wrist.z);
    }
    const maxVal = Math.max(...coords.map(Math.abs)) || 1;
    return coords.map(c => c / maxVal);
}

// Run inference
async function runInference(landmarks) {
    if (!model || !isModelLoaded) return;
    
    try {
        const inputData = processLandmarks(landmarks);
        const inputTensor = tf.tensor2d(inputData, [1, 63]);
        const prediction = model.predict(inputTensor);
        const scores = await prediction.data();
        const maxIdx = scores.indexOf(Math.max(...scores));
        const confidence = scores[maxIdx];
        const label = labelMap[maxIdx];

        if (confidence > 0.75 && label !== "Blank") {
            currentPrediction = label;
            window.currentPrediction = label;
            if (window.updatePrediction) window.updatePrediction(label, confidence);
        } else {
            currentPrediction = "";
            window.currentPrediction = "";
            if (window.updatePrediction) window.updatePrediction("", 0);
        }

        inputTensor.dispose();
        prediction.dispose();
    } catch (error) {
        console.error("Inference error:", error);
    }
}

// Draw landmarks
function drawLandmarkBox(ctx, landmarks, w, h) {
    if (!landmarks || landmarks.length === 0) return;
    
    const x = landmarks.map(l => l.x * w);
    const y = landmarks.map(l => l.y * h);
    const minX = Math.min(...x), maxX = Math.max(...x);
    const minY = Math.min(...y), maxY = Math.max(...y);
    const pad = 20;

    ctx.strokeStyle = "#00c6ff";
    ctx.lineWidth = 2.5;
    ctx.shadowBlur = 8;
    ctx.shadowColor = "#00c6ff";
    ctx.beginPath();
    // Add roundRect if not available
    if (ctx.roundRect) {
        ctx.roundRect(minX - pad, minY - pad, (maxX - minX) + pad * 2, (maxY - minY) + pad * 2, 10);
    } else {
        ctx.rect(minX - pad, minY - pad, (maxX - minX) + pad * 2, (maxY - minY) + pad * 2);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    ctx.fillStyle = "rgba(0,198,255,0.8)";
    landmarks.forEach(l => {
        ctx.beginPath();
        ctx.arc(l.x * w, l.y * h, 4, 0, Math.PI * 2);
        ctx.fill();
    });
}

// Add roundRect if not exists
if (!CanvasRenderingContext2D.prototype.roundRect) {
    CanvasRenderingContext2D.prototype.roundRect = function(x, y, w, h, r) {
        if (w < 2 * r) r = w / 2;
        if (h < 2 * r) r = h / 2;
        this.moveTo(x+r, y);
        this.lineTo(x+w-r, y);
        this.quadraticCurveTo(x+w, y, x+w, y+r);
        this.lineTo(x+w, y+h-r);
        this.quadraticCurveTo(x+w, y+h, x+w-r, y+h);
        this.lineTo(x+r, y+h);
        this.quadraticCurveTo(x, y+h, x, y+h-r);
        this.lineTo(x, y+r);
        this.quadraticCurveTo(x, y, x+r, y);
        return this;
    };
}

// Camera functions
async function startCamera() {
    if (cameraStream) return;
    if (!isMediaPipeReady) {
        console.log("Waiting for MediaPipe to initialize...");
        return;
    }
    
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 } 
        });
        video.srcObject = cameraStream;
        
        video.onloadedmetadata = () => {
            video.play();
            if (!inferenceRunning) {
                inferenceRunning = true;
                predictWebcam();
            }
        };
    } catch (err) {
        console.error("Camera access denied:", err);
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
        cameraStream = null;
        video.srcObject = null;
    }
    inferenceRunning = false;
}

// Webcam prediction loop
async function predictWebcam() {
    if (!inferenceRunning || activeMode !== 'live') return;
    if (!handLandmarker || !isMediaPipeReady) {
        requestAnimationFrame(predictWebcam);
        return;
    }
    
    if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
        liveCanvas.width = video.videoWidth;
        liveCanvas.height = video.videoHeight;
        lastVideoTime = video.currentTime;

        try {
            const result = handLandmarker.detectForVideo(video, performance.now());
            liveCtx.clearRect(0, 0, liveCanvas.width, liveCanvas.height);

            if (result.landmarks && result.landmarks.length > 0) {
                drawLandmarkBox(liveCtx, result.landmarks[0], liveCanvas.width, liveCanvas.height);
                if (isModelLoaded) {
                    await runInference(result.landmarks[0]);
                }
            } else {
                if (window.updatePrediction) window.updatePrediction("", 0);
            }
        } catch (error) {
            console.error("Detection error:", error);
        }
    }
    
    requestAnimationFrame(predictWebcam);
}

// Mode change listener
window.addEventListener('modeChange', async (e) => {
    activeMode = e.detail;
    console.log("Mode changed to:", activeMode);
    
    if (activeMode === 'live') {
        if (isMediaPipeReady) {
            await startCamera();
        } else {
            console.log("Waiting for initialization...");
            // Wait for MediaPipe to be ready
            const checkInterval = setInterval(async () => {
                if (isMediaPipeReady) {
                    clearInterval(checkInterval);
                    await startCamera();
                }
            }, 100);
        }
    } else {
        stopCamera();
        currentPrediction = "";
        window.currentPrediction = "";
        if (window.updatePrediction) window.updatePrediction("", 0);
    }
});

// Image upload handling
const imageUpload = document.getElementById("imageUpload");
if (imageUpload) {
    imageUpload.addEventListener("change", async (event) => {
        const file = event.target.files[0];
        if (!file || !handLandmarker) return;

        const img = new Image();
        img.src = URL.createObjectURL(file);
        img.onload = async () => {
            try {
                await handLandmarker.setOptions({ runningMode: "IMAGE" });
                const result = await handLandmarker.detect(img);
                
                if (result.landmarks && result.landmarks.length > 0) {
                    imageCanvas.width = img.width;
                    imageCanvas.height = img.height;
                    const ctx = imageCanvas.getContext('2d');
                    ctx.drawImage(img, 0, 0);
                    drawLandmarkBox(ctx, result.landmarks[0], imageCanvas.width, imageCanvas.height);
                    if (isModelLoaded) {
                        await runInference(result.landmarks[0]);
                    }
                } else {
                    if (window.updatePrediction) window.updatePrediction("", 0);
                }
                await handLandmarker.setOptions({ runningMode: "VIDEO" });
            } catch (error) {
                console.error("Image detection error:", error);
            }
        };
    });
}



// Initialize everything
initialize();