function pollVideoStatus(videoId, resultUrl) {
    fetch(`/video-status/${videoId}/`)
        .then((response) => response.json())
        .then((data) => {
            console.log('Status fetched:', data.status);
            const statusEl = document.getElementById("status");
            if (statusEl) {
                statusEl.innerText = data.status;
            }

            if (data.status === "completed") {
                window.location.href = resultUrl;
            } else if (data.status === "failed") {
                alert("Processing failed. Please try again.");
            }
        })
        .catch((error) => {
            console.error("Error checking video status:", error);
        });
}

function startPolling(videoId, resultUrl, intervalMs = 5000) {
    setInterval(() => {
        pollVideoStatus(videoId, resultUrl);
    }, intervalMs);
}
