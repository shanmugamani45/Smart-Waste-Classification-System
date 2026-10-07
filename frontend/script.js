// AI-Powered Circular Economy System Client Logic
document.addEventListener("DOMContentLoaded", () => {
    // State variables
    let selectedFile = null;
    let webcamStream = null;
    let currentPredictionId = null;

    // DOM Elements
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const browseBtn = document.getElementById("browse-btn");
    const dropPrompt = document.getElementById("drop-prompt");
    const previewContainer = document.getElementById("preview-container");
    const imagePreview = document.getElementById("image-preview");
    const removeBtn = document.getElementById("remove-btn");

    const webcamToggleBtn = document.getElementById("webcam-toggle-btn");
    const camBtnText = document.getElementById("cam-btn-text");
    const cameraContainer = document.getElementById("camera-container");
    const webcamVideo = document.getElementById("webcam-video");
    const snapBtn = document.getElementById("snap-btn");

    const analyzeBtn = document.getElementById("analyze-btn");
    const resultsEmpty = document.getElementById("results-empty");
    const resultsLoading = document.getElementById("results-loading");
    const resultsContent = document.getElementById("results-content");

    // Result Display Elements
    const resClass = document.getElementById("res-class");
    const resConfidence = document.getElementById("res-confidence");
    const resConfTier = document.getElementById("res-conf-tier");
    const resLatency = document.getElementById("res-latency");
    const resWarning = document.getElementById("res-warning");
    const warningText = document.getElementById("warning-text");
    const resAction = document.getElementById("res-action");
    const resStrategy = document.getElementById("res-strategy");
    const resPriority = document.getElementById("res-priority");
    const resPriorityBadge = document.getElementById("res-priority-badge");
    const resGuidelines = document.getElementById("res-guidelines");
    const resOrigImg = document.getElementById("res-orig-img");
    const resCamImg = document.getElementById("res-cam-img");
    const probBarsContainer = document.getElementById("prob-bars-container");

    // Feedback Elements
    const fbCorrectBtn = document.getElementById("fb-correct-btn");
    const fbIncorrectBtn = document.getElementById("fb-incorrect-btn");
    const feedbackCorrectionPanel = document.getElementById("feedback-correction-panel");
    const correctClassSelect = document.getElementById("correct-class-select");
    const submitCorrectionBtn = document.getElementById("submit-correction-btn");
    const feedbackMsg = document.getElementById("feedback-msg");

    // Dashboard Elements
    const statTotal = document.getElementById("stat-total");
    const statAvgConf = document.getElementById("stat-avg-conf");
    const statDominant = document.getElementById("stat-dominant");
    const statActionDominant = document.getElementById("stat-action-dominant");
    const catDistList = document.getElementById("category-distribution-list");
    const actionDistList = document.getElementById("action-distribution-list");
    const priorityDistList = document.getElementById("priority-distribution-list");

    const historyTbody = document.getElementById("history-tbody");
    const comparisonTbody = document.getElementById("comparison-tbody");
    const specActiveModel = document.getElementById("spec-active-model");
    const statusIndicator = document.getElementById("status-indicator");
    const statusText = document.getElementById("status-text");

    // -------------------------------------------------------------
    // Initial Setup & Health Check
    // -------------------------------------------------------------
    checkSystemHealth();
    loadDashboardStats();
    loadHistory();
    loadModelSpecs();

    async function checkSystemHealth() {
        try {
            const res = await fetch("/health");
            if (res.ok) {
                const data = await res.json();
                statusIndicator.className = "status-dot online";
                statusText.innerText = `Online (${data.model_name})`;
                specActiveModel.innerText = data.model_name;
            } else {
                statusIndicator.className = "status-dot";
                statusText.innerText = "Degraded";
            }
        } catch (e) {
            statusIndicator.className = "status-dot";
            statusText.innerText = "Offline";
        }
    }

    // -------------------------------------------------------------
    // File Ingestion & Drag and Drop
    // -------------------------------------------------------------
    browseBtn.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    function handleFileSelect(file) {
        if (!file.type.startsWith("image/")) {
            alert("Please select a valid image file (JPG, PNG, WEBP).");
            return;
        }
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropPrompt.style.display = "none";
            previewContainer.style.display = "block";
            cameraContainer.style.display = "none";
            stopWebcam();
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    removeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        resetFileInput();
    });

    function resetFileInput() {
        selectedFile = null;
        fileInput.value = "";
        imagePreview.src = "";
        previewContainer.style.display = "none";
        dropPrompt.style.display = "flex";
        analyzeBtn.disabled = true;
    }

    // -------------------------------------------------------------
    // Live Webcam Stream Handler
    // -------------------------------------------------------------
    webcamToggleBtn.addEventListener("click", async () => {
        if (webcamStream) {
            stopWebcam();
            cameraContainer.style.display = "none";
            dropPrompt.style.display = "flex";
            camBtnText.innerText = "📷 Use Camera";
        } else {
            try {
                webcamStream = await navigator.mediaDevices.getUserMedia({
                    video: { width: { ideal: 640 }, height: { ideal: 480 } }
                });
                webcamVideo.srcObject = webcamStream;
                cameraContainer.style.display = "block";
                dropPrompt.style.display = "none";
                previewContainer.style.display = "none";
                camBtnText.innerText = "✕ Close Camera";
            } catch (err) {
                alert("Unable to access webcam: " + err.message);
            }
        }
    });

    snapBtn.addEventListener("click", () => {
        if (!webcamVideo.videoWidth) return;
        const canvas = document.createElement("canvas");
        canvas.width = webcamVideo.videoWidth;
        canvas.height = webcamVideo.videoHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(webcamVideo, 0, 0, canvas.width, canvas.height);

        canvas.toBlob((blob) => {
            selectedFile = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
            imagePreview.src = canvas.toDataURL("image/jpeg");
            previewContainer.style.display = "block";
            cameraContainer.style.display = "none";
            stopWebcam();
            camBtnText.innerText = "📷 Use Camera";
            analyzeBtn.disabled = false;
        }, "image/jpeg", 0.95);
    });

    function stopWebcam() {
        if (webcamStream) {
            webcamStream.getTracks().forEach((track) => track.stop());
            webcamStream = null;
        }
    }

    // -------------------------------------------------------------
    // Prediction Workflow (POST /predict)
    // -------------------------------------------------------------
    analyzeBtn.addEventListener("click", async () => {
        if (!selectedFile) return;

        resultsEmpty.style.display = "none";
        resultsContent.style.display = "none";
        resultsLoading.style.display = "block";
        analyzeBtn.disabled = true;

        const formData = new FormData();
        formData.append("file", selectedFile);

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || "Prediction request failed.");
            }

            const data = await response.json();
            renderPredictionResults(data);
            loadDashboardStats();
            loadHistory();
        } catch (error) {
            alert("Analysis failed: " + error.message);
            resultsEmpty.style.display = "block";
        } finally {
            resultsLoading.style.display = "none";
            analyzeBtn.disabled = false;
        }
    });

    function renderPredictionResults(data) {
        currentPredictionId = data.id;
        const pred = data.prediction;
        const rec = data.recommendation;
        const exp = data.explainability;

        resClass.innerText = pred.class.toUpperCase();
        resConfidence.innerText = (pred.confidence * 100).toFixed(1) + "%";
        resLatency.innerText = pred.latency_ms + " ms";

        // Confidence badge styling
        resConfTier.innerText = rec.confidence_level;
        resConfTier.className = `conf-badge ${rec.confidence_level}`;

        // Warning or Verified Banner
        if (rec.confidence_warning) {
            warningText.innerText = rec.confidence_warning;
            resWarning.style.display = "block";
            if (rec.confidence_warning.includes("Verified by")) {
                resWarning.style.background = "rgba(16, 185, 129, 0.15)";
                resWarning.style.borderColor = "rgba(16, 185, 129, 0.4)";
                resWarning.style.color = "#34d399";
            } else {
                resWarning.style.background = "rgba(245, 158, 11, 0.1)";
                resWarning.style.borderColor = "rgba(245, 158, 11, 0.3)";
                resWarning.style.color = "#fde047";
            }
        } else {
            resWarning.style.display = "none";
        }

        // Circular Policy Values
        resAction.innerText = rec.recommended_action;
        resStrategy.innerText = rec.circular_strategy;
        resPriority.innerText = rec.environmental_priority;
        resPriorityBadge.className = `priority-indicator priority-${rec.environmental_priority}`;
        resGuidelines.innerText = rec.guidelines;

        // Grad-CAM Heatmap
        resOrigImg.src = imagePreview.src;
        if (exp.gradcam_available && exp.heatmap_overlay) {
            resCamImg.src = exp.heatmap_overlay;
        } else {
            resCamImg.src = imagePreview.src;
        }

        // Probability Distribution Bars
        renderProbabilityBars(pred.all_probabilities, pred.class);

        // Reset Feedback Panel
        feedbackCorrectionPanel.style.display = "none";
        feedbackMsg.style.display = "none";
        fbCorrectBtn.disabled = false;
        fbIncorrectBtn.disabled = false;

        resultsContent.style.display = "block";
    }

    function renderProbabilityBars(probs, predictedClass) {
        probBarsContainer.innerHTML = "";
        const sortedEntries = Object.entries(probs).sort((a, b) => b[1] - a[1]);

        sortedEntries.forEach(([clsName, prob]) => {
            const pct = (prob * 100).toFixed(1);
            const isTop = clsName.toLowerCase() === predictedClass.toLowerCase();

            const item = document.createElement("div");
            item.className = "prob-item";
            item.innerHTML = `
                <div class="prob-info">
                    <span style="${isTop ? 'font-weight: 700; color: #10b981;' : ''}">
                        ${clsName.charAt(0).toUpperCase() + clsName.slice(1)}
                    </span>
                    <span>${pct}%</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill" style="width: ${pct}%;"></div>
                </div>
            `;
            probBarsContainer.appendChild(item);
        });
    }

    // -------------------------------------------------------------
    // Feedback Workflow (POST /feedback)
    // -------------------------------------------------------------
    fbCorrectBtn.addEventListener("click", () => sendFeedback("correct", null));
    fbIncorrectBtn.addEventListener("click", () => {
        feedbackCorrectionPanel.style.display = "block";
    });

    submitCorrectionBtn.addEventListener("click", () => {
        const corrected = correctClassSelect.value;
        sendFeedback("incorrect", corrected);
    });

    async function sendFeedback(feedbackType, correctedClass) {
        if (!currentPredictionId) return;

        try {
            const res = await fetch("/feedback", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    prediction_id: currentPredictionId,
                    user_feedback: feedbackType,
                    corrected_class: correctedClass
                })
            });

            if (res.ok) {
                const data = await res.json();
                feedbackMsg.style.display = "block";
                feedbackMsg.innerText = data.message || "Feedback recorded! The system has actively learned this sample.";
                fbCorrectBtn.disabled = true;
                fbIncorrectBtn.disabled = true;
                feedbackCorrectionPanel.style.display = "none";

                // Instant UI update to reflect the corrected label
                if (correctedClass) {
                    resClass.innerText = correctedClass.toUpperCase();
                    resConfidence.innerText = "99.9%";
                    warningText.innerText = "Verified by Prior User Feedback";
                    resWarning.style.display = "block";
                    resWarning.style.background = "rgba(16, 185, 129, 0.15)";
                    resWarning.style.borderColor = "rgba(16, 185, 129, 0.4)";
                    resWarning.style.color = "#34d399";
                }
                loadDashboardStats();
                loadHistory();
            } else {
                const errData = await res.json();
                alert("Failed to submit feedback: " + (errData.detail || "Server error"));
            }
        } catch (e) {
            alert("Failed to submit feedback: " + e.message);
        }
    }

    // -------------------------------------------------------------
    // Dashboard Stats (GET /statistics)
    // -------------------------------------------------------------
    async function loadDashboardStats() {
        try {
            const res = await fetch("/statistics");
            if (!res.ok) return;
            const data = await res.json();

            statTotal.innerText = data.total_predictions;
            statAvgConf.innerText = (data.average_confidence * 100).toFixed(1) + "%";

            // Find dominant category & action
            let topCat = "-", topCatCount = -1;
            for (const [k, v] of Object.entries(data.category_distribution || {})) {
                if (v > topCatCount) { topCatCount = v; topCat = k; }
            }
            statDominant.innerText = topCat !== "-" ? topCat.toUpperCase() : "None";

            let topAction = "-", topActionCount = -1;
            for (const [k, v] of Object.entries(data.action_distribution || {})) {
                if (v > topActionCount) { topActionCount = v; topAction = k; }
            }
            statActionDominant.innerText = topAction !== "-" ? topAction : "None";

            // Category breakdown
            catDistList.innerHTML = "";
            const catEntries = Object.entries(data.category_percentages || {});
            if (catEntries.length === 0) {
                catDistList.innerHTML = `<span class="text-muted" style="font-size:0.85rem;">No predictions yet.</span>`;
            } else {
                catEntries.forEach(([cat, pct]) => {
                    const count = data.category_distribution[cat] || 0;
                    catDistList.innerHTML += `
                        <div class="dist-row">
                            <div class="dist-meta">
                                <span>${cat.charAt(0).toUpperCase() + cat.slice(1)} (${count})</span>
                                <span>${pct}%</span>
                            </div>
                            <div class="dist-bar-track">
                                <div class="dist-bar-fill" style="width: ${pct}%;"></div>
                            </div>
                        </div>
                    `;
                });
            }

            // Action breakdown
            actionDistList.innerHTML = "";
            const totalPreds = data.total_predictions || 1;
            const actionEntries = Object.entries(data.action_distribution || {});
            if (actionEntries.length === 0) {
                actionDistList.innerHTML = `<span class="text-muted" style="font-size:0.85rem;">No actions recorded yet.</span>`;
            } else {
                actionEntries.forEach(([act, count]) => {
                    const pct = ((count / totalPreds) * 100).toFixed(1);
                    actionDistList.innerHTML += `
                        <div class="dist-row">
                            <div class="dist-meta">
                                <span>${act} (${count})</span>
                                <span>${pct}%</span>
                            </div>
                            <div class="dist-bar-track">
                                <div class="dist-bar-fill" style="width: ${pct}%; background: #38bdf8;"></div>
                            </div>
                        </div>
                    `;
                });
            }

            // Priority breakdown
            priorityDistList.innerHTML = "";
            const priorityEntries = Object.entries(data.priority_distribution || {});
            if (priorityEntries.length === 0) {
                priorityDistList.innerHTML = `<span class="text-muted" style="font-size:0.85rem;">No data yet.</span>`;
            } else {
                priorityEntries.forEach(([prio, count]) => {
                    const pct = ((count / totalPreds) * 100).toFixed(1);
                    const color = prio === "HIGH" ? "#ef4444" : prio === "MEDIUM" ? "#f59e0b" : "#38bdf8";
                    priorityDistList.innerHTML += `
                        <div class="dist-row">
                            <div class="dist-meta">
                                <span>${prio} (${count})</span>
                                <span>${pct}%</span>
                            </div>
                            <div class="dist-bar-track">
                                <div class="dist-bar-fill" style="width: ${pct}%; background: ${color};"></div>
                            </div>
                        </div>
                    `;
                });
            }
        } catch (e) {
            console.error("Dashboard statistics loading failed:", e);
        }
    }

    // -------------------------------------------------------------
    // History Table (GET /history)
    // -------------------------------------------------------------
    async function loadHistory() {
        try {
            const res = await fetch("/history?limit=15");
            if (!res.ok) return;
            const history = await res.json();

            if (history.length === 0) {
                historyTbody.innerHTML = `<tr><td colspan="8" class="text-center">No predictions recorded yet.</td></tr>`;
                return;
            }

            historyTbody.innerHTML = history.map((item) => {
                const confPct = (item.confidence * 100).toFixed(1) + "%";
                const feedbackBadge = item.user_feedback === "correct" 
                    ? `<span style="color:#34d399;">✓ Verified</span>` 
                    : item.user_feedback === "incorrect" 
                    ? `<span style="color:#f87171;">✗ Corrected (${item.corrected_class})</span>` 
                    : `<span style="color:#94a3b8;">Pending</span>`;

                return `
                    <tr>
                        <td>#${item.id}</td>
                        <td>${new Date(item.timestamp).toLocaleString()}</td>
                        <td><strong>${item.predicted_class}</strong></td>
                        <td>${confPct}</td>
                        <td>${item.recommended_action}</td>
                        <td>${item.circular_strategy}</td>
                        <td><span class="priority-indicator priority-${item.environmental_priority}">${item.environmental_priority}</span></td>
                        <td>${feedbackBadge}</td>
                    </tr>
                `;
            }).join("");
        } catch (e) {
            console.error("History loading error:", e);
        }
    }

    // -------------------------------------------------------------
    // Model Specifications & Benchmarks (GET /model-info)
    // -------------------------------------------------------------
    async function loadModelSpecs() {
        try {
            const res = await fetch("/model-info");
            if (!res.ok) return;
            const data = await res.json();

            specActiveModel.innerText = data.active_model;

            const comparison = data.model_comparison || [];
            if (comparison.length === 0) {
                comparisonTbody.innerHTML = `<tr><td colspan="8" class="text-center">No comparison models loaded.</td></tr>`;
                return;
            }

            comparisonTbody.innerHTML = comparison.map((m) => `
                <tr>
                    <td><strong>${m.Model}</strong></td>
                    <td>${m.Accuracy}</td>
                    <td>${m.Precision}</td>
                    <td>${m.Recall}</td>
                    <td>${m.F1}</td>
                    <td>${m.Parameters}</td>
                    <td>${m.Size_MB}</td>
                    <td>${m.Inference_Time_MS}</td>
                </tr>
            `).join("");
        } catch (e) {
            console.error("Specs loading error:", e);
        }
    }
});
