/**
 * VISTA Studio Frontend Logic (app.js)
 * 交互控制：主题切换、YAML默认参数同步、预设联动、滑块实时显示、异步处理、多看板视图与图层画廊。
 */

document.addEventListener("DOMContentLoaded", () => {
    // -------------------------------------------------------------
    // Lucide Icons Initialization
    // -------------------------------------------------------------
    if (window.lucide) {
        window.lucide.createIcons();
    }

    // -------------------------------------------------------------
    // Theme Switcher (Dark / Light)
    // -------------------------------------------------------------
    const themeToggleBtn = document.getElementById("theme-toggle");
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const savedTheme = localStorage.getItem("vista-theme") || (prefersDark ? "dark" : "light");

    if (savedTheme === "dark") {
        document.body.classList.add("dark");
    }

    themeToggleBtn.addEventListener("click", () => {
        document.body.classList.toggle("dark");
        const currentTheme = document.body.classList.contains("dark") ? "dark" : "light";
        localStorage.setItem("vista-theme", currentTheme);
        if (window.lucide) window.lucide.createIcons();
    });

    // -------------------------------------------------------------
    // File Upload & Drag-and-Drop
    // -------------------------------------------------------------
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    const previewBox = document.getElementById("preview-box");
    const imagePreview = document.getElementById("image-preview");
    const fileNameEl = document.getElementById("file-name");
    const fileMetaEl = document.getElementById("file-meta");
    const removeFileBtn = document.getElementById("remove-file-btn");

    const resultTargetImg = document.getElementById("result-target-img");
    const targetPlaceholder = document.getElementById("target-placeholder");
    const targetDimBadge = document.getElementById("target-dim");

    let currentFile = null;

    function handleFile(file) {
        if (!file || !file.type.startsWith("image/")) {
            alert("请选择有效的图像文件 (PNG / JPG / WEBP / BMP)");
            return;
        }

        currentFile = file;
        fileNameEl.textContent = file.name;

        const reader = new FileReader();
        reader.onload = (e) => {
            const dataUrl = e.target.result;
            imagePreview.src = dataUrl;
            resultTargetImg.src = dataUrl;

            // 获取图像实际分辨率
            const img = new Image();
            img.onload = () => {
                const dimStr = `${img.naturalWidth} × ${img.naturalHeight}`;
                fileMetaEl.textContent = `${dimStr} (${(file.size / 1024).toFixed(1)} KB)`;
                targetDimBadge.textContent = dimStr;
            };
            img.src = dataUrl;

            dropzone.classList.add("hidden");
            previewBox.classList.remove("hidden");
            targetPlaceholder.classList.add("hidden");
            resultTargetImg.classList.remove("hidden");
        };
        reader.readAsDataURL(file);
    }

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    });

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    removeFileBtn.addEventListener("click", () => {
        currentFile = null;
        fileInput.value = "";
        previewBox.classList.add("hidden");
        dropzone.classList.remove("hidden");
        resultTargetImg.src = "";
        resultTargetImg.classList.add("hidden");
        targetPlaceholder.classList.remove("hidden");
        targetDimBadge.textContent = "--";
    });

    // -------------------------------------------------------------
    // Live Slider Value Display
    // -------------------------------------------------------------
    const sliderBindings = [
        { id: "target_size", valId: "target_size_val", format: (v) => (v === "0" ? "原图 (0)" : `${v} px`) },
        { id: "slic_n_segments", valId: "slic_n_segments_val", format: (v) => v },
        { id: "slic_compactness", valId: "slic_compactness_val", format: (v) => parseFloat(v).toFixed(1) },
        { id: "dbscan_eps", valId: "dbscan_eps_val", format: (v) => parseFloat(v).toFixed(1) },
        { id: "pred_iou_thresh", valId: "pred_iou_thresh_val", format: (v) => parseFloat(v).toFixed(2) },
        { id: "stability_score_thresh", valId: "stability_score_thresh_val", format: (v) => parseFloat(v).toFixed(2) },
        { id: "points_per_side", valId: "points_per_side_val", format: (v) => v },
        { id: "min_area", valId: "min_area_val", format: (v) => parseFloat(v).toFixed(5) },
        { id: "iou_sam_slic_thresh", valId: "iou_sam_slic_thresh_val", format: (v) => parseFloat(v).toFixed(2) },
        { id: "iou_sam_internal_thresh", valId: "iou_sam_internal_thresh_val", format: (v) => parseFloat(v).toFixed(2) },
        { id: "parent_contain_thresh", valId: "parent_contain_thresh_val", format: (v) => parseFloat(v).toFixed(2) },
        { id: "self_pure_std_thresh", valId: "self_pure_std_thresh_val", format: (v) => parseFloat(v).toFixed(1) },
        { id: "parent_pure_std_thresh", valId: "parent_pure_std_thresh_val", format: (v) => parseFloat(v).toFixed(1) },
        { id: "color_diff_thresh", valId: "color_diff_thresh_val", format: (v) => parseFloat(v).toFixed(1) },
        { id: "bzer_max_error", valId: "bzer_max_error_val", format: (v) => parseFloat(v).toFixed(4) },
        { id: "line_threshold", valId: "line_threshold_val", format: (v) => parseFloat(v).toFixed(4) },
        { id: "learning_rate", valId: "learning_rate_val", format: (v) => parseFloat(v).toFixed(2) },
        { id: "num_iters", valId: "num_iters_val", format: (v) => v },
        { id: "early_stopping_patience", valId: "early_stopping_patience_val", format: (v) => v },
        { id: "rm_color_threshold", valId: "rm_color_threshold_val", format: (v) => parseFloat(v).toFixed(3) },
        { id: "refine_iters", valId: "refine_iters_val", format: (v) => v },
    ];

    function updateSliderDisplay(item) {
        const el = document.getElementById(item.id);
        const valEl = document.getElementById(item.valId);
        if (el && valEl) {
            valEl.textContent = item.format(el.value);
        }
    }

    sliderBindings.forEach((item) => {
        const el = document.getElementById(item.id);
        if (el) {
            el.addEventListener("input", () => updateSliderDisplay(item));
            updateSliderDisplay(item);
        }
    });

    // -------------------------------------------------------------
    // Load & Apply Default YAML Configuration
    // -------------------------------------------------------------
    async function loadYamlDefaults() {
        try {
            const resp = await fetch("/config/defaults");
            if (!resp.ok) return;
            const defaults = await resp.json();

            Object.keys(defaults).forEach((key) => {
                const el = document.getElementById(key);
                if (el) {
                    if (el.type === "checkbox") {
                        el.checked = Boolean(defaults[key]);
                    } else {
                        el.value = defaults[key];
                    }
                }
            });

            sliderBindings.forEach(updateSliderDisplay);
        } catch (e) {
            console.warn("读取 /config/defaults 失败:", e);
        }
    }

    // 初始加载 YAML 默认配置
    loadYamlDefaults();

    // -------------------------------------------------------------
    // Quick Presets
    // -------------------------------------------------------------
    const presets = {
        fast: {
            target_size: 512,
            slic_n_segments: 1000,
            slic_compactness: 5.0,
            dbscan_eps: 6.0,
            use_sam: true,
            pred_iou_thresh: 0.85,
            stability_score_thresh: 0.90,
            points_per_side: 32,
            min_area: 0.0003,
            bzer_max_error: 2.0,
            line_threshold: 2.0,
            learning_rate: 0.15,
            num_iters: 300,
            early_stopping_patience: 15,
            refine_iters: 40,
        },
        balanced: {
            target_size: 0,
            slic_n_segments: 2000,
            slic_compactness: 5.0,
            dbscan_eps: 5.0,
            use_sam: true,
            pred_iou_thresh: 0.88,
            stability_score_thresh: 0.95,
            points_per_side: 64,
            min_area: 0.00015,
            bzer_max_error: 1.5,
            line_threshold: 2.0,
            learning_rate: 0.10,
            num_iters: 1000,
            early_stopping_patience: 20,
            refine_iters: 80,
        },
        high_quality: {
            target_size: 0,
            slic_n_segments: 3000,
            slic_compactness: 4.0,
            dbscan_eps: 4.0,
            use_sam: true,
            pred_iou_thresh: 0.92,
            stability_score_thresh: 0.97,
            points_per_side: 64,
            min_area: 0.0001,
            bzer_max_error: 1.0,
            line_threshold: 1.0,
            learning_rate: 0.08,
            num_iters: 1500,
            early_stopping_patience: 30,
            refine_iters: 120,
        },
    };

    function applyPreset(presetKey) {
        const p = presets[presetKey];
        if (!p) return;

        Object.keys(p).forEach((key) => {
            const el = document.getElementById(key);
            if (el) {
                if (el.type === "checkbox") {
                    el.checked = Boolean(p[key]);
                } else {
                    el.value = p[key];
                }
            }
        });

        sliderBindings.forEach(updateSliderDisplay);

        document.querySelectorAll(".preset-btn").forEach((btn) => {
            btn.classList.toggle("active", btn.getAttribute("data-preset") === presetKey);
        });
    }

    document.querySelectorAll(".preset-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            const presetKey = btn.getAttribute("data-preset");
            applyPreset(presetKey);
        });
    });

    document.getElementById("reset-params-btn").addEventListener("click", () => {
        loadYamlDefaults();
        document.querySelectorAll(".preset-btn").forEach((btn) => {
            btn.classList.toggle("active", btn.getAttribute("data-preset") === "balanced");
        });
    });

    // -------------------------------------------------------------
    // Parameter Step Navigation Tabs (1. 预处理 / 2. SLIC / 3. SAM / 4. 融合 / 5. DiffVG / 全部)
    // -------------------------------------------------------------
    const paramNavBtns = document.querySelectorAll(".param-nav-btn");
    const paramSections = document.querySelectorAll(".param-section");

    paramNavBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
            const step = btn.getAttribute("data-step");

            paramNavBtns.forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");

            paramSections.forEach((sec) => {
                if (step === "all" || sec.getAttribute("data-section") === step) {
                    sec.classList.remove("hidden");
                } else {
                    sec.classList.add("hidden");
                }
            });
        });
    });

    // -------------------------------------------------------------
    // Main Workspace Tabs (SVG / GIF / Stages / Layers)
    // -------------------------------------------------------------
    const tabButtons = document.querySelectorAll(".tab-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");

    tabButtons.forEach((btn) => {
        btn.addEventListener("click", () => {
            const targetId = btn.getAttribute("data-tab");

            tabButtons.forEach((b) => b.classList.remove("active"));
            tabPanes.forEach((p) => {
                p.classList.remove("active");
                p.classList.add("hidden");
            });

            btn.classList.add("active");
            const targetPane = document.getElementById(targetId);
            if (targetPane) {
                targetPane.classList.remove("hidden");
                targetPane.classList.add("active");
            }
        });
    });

    // -------------------------------------------------------------
    // API Execution (/process)
    // -------------------------------------------------------------
    const processBtn = document.getElementById("process-btn");
    const loadingOverlay = document.getElementById("loading-overlay");

    const svgContainer = document.getElementById("svg-container");
    const svgPlaceholder = document.getElementById("svg-placeholder");
    const downloadSvgBtn = document.getElementById("download-svg-btn");

    const animationGif = document.getElementById("animation-gif");
    const gifPlaceholder = document.getElementById("gif-placeholder");
    const downloadGifBtn = document.getElementById("download-gif-btn");

    // Overviews
    const ovSlic = document.getElementById("ov-slic");
    const ovSam = document.getElementById("ov-sam");
    const ovOrigin = document.getElementById("ov-origin");
    const ovPre = document.getElementById("ov-pre");

    // Layers Gallery
    const layerGallery = document.getElementById("layer-gallery");
    const galleryCount = document.getElementById("gallery-count");

    // Stats
    const statTime = document.getElementById("stat-time");
    const statSegTime = document.getElementById("stat-seg-time");
    const statVecTime = document.getElementById("stat-vec-time");
    const statShapes = document.getElementById("stat-shapes");
    const statPoints = document.getElementById("stat-points");
    const statLoss = document.getElementById("stat-loss");

    let currentSvgUrl = "";
    let currentGifUrl = "";

    processBtn.addEventListener("click", async () => {
        if (!currentFile) {
            alert("请先上传目标图像！");
            return;
        }

        const formData = new FormData();
        formData.append("file", currentFile);

        // 收集全部超参数
        const paramKeys = [
            "target_size", "slic_n_segments", "slic_compactness", "dbscan_eps",
            "pred_iou_thresh", "stability_score_thresh", "points_per_side",
            "min_area", "iou_sam_slic_thresh", "iou_sam_internal_thresh", "parent_contain_thresh",
            "self_pure_std_thresh", "parent_pure_std_thresh", "color_diff_thresh",
            "bzer_max_error", "line_threshold", "learning_rate", "num_iters",
            "early_stopping_patience", "rm_color_threshold", "refine_iters"
        ];

        paramKeys.forEach((key) => {
            const el = document.getElementById(key);
            if (el) {
                formData.append(key, el.value);
            }
        });

        // 收集复选框开关
        formData.append("use_sam", document.getElementById("use_sam").checked ? "true" : "false");
        formData.append("prune_enabled", document.getElementById("prune_enabled").checked ? "true" : "false");
        formData.append("is_stroke", document.getElementById("is_stroke").checked ? "true" : "false");
        formData.append("crop_n_layers", "1");
        formData.append("color_lr", "0.01");
        formData.append("collinear_scale", "0.01");

        // 开启加载遮罩
        loadingOverlay.classList.remove("hidden");
        processBtn.disabled = true;

        try {
            const resp = await fetch("/process", {
                method: "POST",
                body: formData,
            });

            if (!resp.ok) {
                const errData = await resp.json().catch(() => ({}));
                throw new Error(errData.detail || `请求失败 (${resp.status})`);
            }

            const data = await resp.json();

            // 1. 渲染最终矢量 SVG (解析并注入自适应 viewBox，确保全景无裁剪缩放)
            currentSvgUrl = data.svg_url;
            if (currentSvgUrl) {
                const svgResp = await fetch(currentSvgUrl);
                const svgText = await svgResp.text();

                const parser = new DOMParser();
                const svgDoc = parser.parseFromString(svgText, "image/svg+xml");
                const svgEl = svgDoc.querySelector("svg");

                if (svgEl) {
                    const w = svgEl.getAttribute("width") || "512";
                    const h = svgEl.getAttribute("height") || "512";
                    if (!svgEl.getAttribute("viewBox")) {
                        const numW = parseFloat(w) || 512;
                        const numH = parseFloat(h) || 512;
                        svgEl.setAttribute("viewBox", `0 0 ${numW} ${numH}`);
                    }
                    svgEl.setAttribute("width", "100%");
                    svgEl.setAttribute("height", "100%");
                    svgEl.setAttribute("preserveAspectRatio", "xMidYMid meet");
                    svgEl.style.maxWidth = "100%";
                    svgEl.style.maxHeight = "420px";
                    svgEl.style.display = "block";

                    svgContainer.innerHTML = "";
                    svgContainer.appendChild(svgEl);
                } else {
                    svgContainer.innerHTML = svgText;
                }

                svgPlaceholder.classList.add("hidden");
                downloadSvgBtn.disabled = false;
            }

            // 2. 渲染优化动画 GIF
            currentGifUrl = data.gif_url;
            if (currentGifUrl) {
                animationGif.src = `${currentGifUrl}?t=${Date.now()}`;
                animationGif.classList.remove("hidden");
                gifPlaceholder.classList.add("hidden");
                downloadGifBtn.disabled = false;
            }

            // 3. 渲染全阶段看板图片
            const ov = data.overviews || {};
            if (ov.slic) { ovSlic.src = `${ov.slic}?t=${Date.now()}`; ovSlic.classList.remove("hidden"); const e = document.getElementById("ov-slic-empty"); if (e) e.classList.add("hidden"); }
            if (ov.sam) { ovSam.src = `${ov.sam}?t=${Date.now()}`; ovSam.classList.remove("hidden"); const e = document.getElementById("ov-sam-empty"); if (e) e.classList.add("hidden"); }
            if (ov.origin) { ovOrigin.src = `${ov.origin}?t=${Date.now()}`; ovOrigin.classList.remove("hidden"); const e = document.getElementById("ov-origin-empty"); if (e) e.classList.add("hidden"); }
            if (ov.pre) { ovPre.src = `${ov.pre}?t=${Date.now()}`; ovPre.classList.remove("hidden"); const e = document.getElementById("ov-pre-empty"); if (e) e.classList.add("hidden"); }

            // 4. 渲染图层分解画廊
            const layers = data.layers || [];
            galleryCount.textContent = `共 ${layers.length} 个独立纯色图层 (pre_masks)`;
            layerGallery.innerHTML = "";

            if (layers.length > 0) {
                layers.forEach((layer, idx) => {
                    const card = document.createElement("div");
                    card.className = "layer-item-card";
                    card.innerHTML = `
                        <img src="${layer.colored_url}" alt="${layer.name}" class="layer-item-img">
                        <div class="layer-item-meta">
                            <span>#${idx} ${layer.name.replace('.png', '')}</span>
                        </div>
                    `;
                    layerGallery.appendChild(card);
                });
            } else {
                layerGallery.innerHTML = `<div class="gallery-placeholder">未生成图层</div>`;
            }

            // 5. 更新指标看板
            const stats = data.stats || {};
            statTime.textContent = `${(stats.total_time_sec || 0).toFixed(2)} s`;
            statSegTime.textContent = `${(stats.segment_time_sec || 0).toFixed(2)} s`;
            statVecTime.textContent = `${(stats.vectorize_time_sec || 0).toFixed(2)} s`;
            statShapes.textContent = stats.shapes || "--";
            statPoints.textContent = stats.path_point_nums || "--";
            statLoss.textContent = (stats.mse_loss !== undefined) ? stats.mse_loss.toFixed(6) : "--";

        } catch (err) {
            console.error("Vectorization Error:", err);
            alert(`矢量化执行出错: ${err.message}`);
        } finally {
            loadingOverlay.classList.add("hidden");
            processBtn.disabled = false;
        }
    });

    // -------------------------------------------------------------
    // GIF Click to Replay
    // -------------------------------------------------------------
    animationGif.addEventListener("click", () => {
        if (currentGifUrl) {
            animationGif.src = "";
            setTimeout(() => {
                animationGif.src = `${currentGifUrl}?t=${Date.now()}`;
            }, 50);
        }
    });

    // -------------------------------------------------------------
    // Download Buttons
    // -------------------------------------------------------------
    downloadSvgBtn.addEventListener("click", () => {
        if (currentSvgUrl) {
            window.location.href = `/download?file_url=${encodeURIComponent(currentSvgUrl)}`;
        }
    });

    downloadGifBtn.addEventListener("click", () => {
        if (currentGifUrl) {
            window.location.href = `/download?file_url=${encodeURIComponent(currentGifUrl)}`;
        }
    });
});