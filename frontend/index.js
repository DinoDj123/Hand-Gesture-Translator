const startCameraButton = document.getElementById("start_camera");
const predictButton = document.getElementById("predict");
const stopCameraButton = document.getElementById("stop_camera");

const video = document.querySelector("video");
const status = document.getElementById("status");
const result = document.getElementById("result");

const constraints = {
    video: true,
    audio: false
};

let stream = null;

startCameraButton.addEventListener("click", async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;
        video.onloadedmetadata = () => video.play();

        status.textContent = "Camera started";
    } catch (err) {
        console.error(`${err.name}: ${err.message}`);
        status.textContent = "Could not start camera";
    }
});

stopCameraButton.addEventListener("click", () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
        video.srcObject = null;
        status.textContent = "Camera stopped";
    }
});

predictButton.addEventListener("click", async () => {
    if (!stream) {
        status.textContent = "Start the camera first";
        return;
    }

    status.textContent = "Recording...";

    const recordedBlob = await recordVideo(stream, 3000);

    status.textContent = "Sending video to backend...";

    const formData = new FormData();
    formData.append("file", recordedBlob, "gesture.webm");

    try {
        const response = await fetch("http://localhost:8000/predict/", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error("Backend request failed");
        }

        const data = await response.json();

        result.textContent = `Prediction: ${data.prediction}`;
        status.textContent = "";
    } catch (err) {
        console.error(err);
        result.textContent = "Prediction failed";
        status.textContent = "";
    }
});

function recordVideo(stream, durationMs) {
    return new Promise((resolve) => {
        const chunks = [];
        const recorder = new MediaRecorder(stream);

        recorder.ondataavailable = event => {
            if (event.data.size > 0) {
                chunks.push(event.data);
            }
        };

        recorder.onstop = () => {
            const blob = new Blob(chunks, {
                type: "video/webm"
            });

            resolve(blob);
        };

        recorder.start();

        setTimeout(() => {
            recorder.stop();
        }, durationMs);
    });
}
