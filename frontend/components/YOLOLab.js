import { ref, computed, onMounted, watch } from 'vue';
import ImageSource from './ImageSource.js';

export default {
    name: 'YOLOLab',
    components: {
        ImageSource
    },
    template: `
    <div class="yolo-lab">
        <div class="lab-header" style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
            <h2>👁️ YOLO 视觉实验室</h2>
            <div class="tabs" style="margin-bottom:0; border-bottom:none;">
                <button :class="{active: activeSubTab === 'detect'}" @click="activeSubTab = 'detect'">目标检测</button>
                <button :class="{active: activeSubTab === 'segment'}" @click="activeSubTab = 'segment'">图像分割</button>
                <button :class="{active: activeSubTab === 'pose'}" @click="activeSubTab = 'pose'">姿态估计</button>
                <button :class="{active: activeSubTab === 'track'}" @click="activeSubTab = 'track'">目标追踪</button>
            </div>
        </div>

        <div class="layout-grid">
            <div class="sidebar-panel">
                <div>
                    <h3>1. 图像源</h3>
                    <p v-if="activeSubTab === 'track'" style="font-size:13px; color:var(--text-muted); margin-bottom: 8px;">追踪功能已整合到实时检测中。只需开启视频或流，系统将自动进行连续探测与分析。</p>
                    <ImageSource :hide-capture-btn="true" @image-selected="onImageSelected" @stream-frame="onStreamFrame" />
                    <hr>
                    
                    <h3>2. 设置参数</h3>
                    <div class="control-group">
                        <label>模型选择</label>
                        <select v-model="modelName">
                            <option v-for="m in availableModels" :key="m.value" :value="m.value">{{ m.label }}</option>
                        </select>
                    </div>

                    <div class="control-group">
                        <label>置信度阈值 (Conf): {{ conf }}</label>
                        <input type="range" v-model="conf" min="0.05" max="1.0" step="0.05">
                        <label style="margin-top:8px;">交并比阈值 (IoU): {{ iou }}</label>
                        <input type="range" v-model="iou" min="0.05" max="0.95" step="0.05">
                    </div>

                    <div class="control-group checkbox" style="display:flex; flex-direction:column; gap:8px;">
                        <label><input type="checkbox" v-model="showLabels"> 显示目标标签</label>
                        <label><input type="checkbox" v-model="showBoxes"> 显示目标框</label>
                        <label><input type="checkbox" v-model="showConfText"> 显示置信度</label>
                        <label v-if="['segment', 'track'].includes(activeSubTab)"><input type="checkbox" v-model="showMasks"> 显示掩码</label>
                        <label v-if="['segment', 'track'].includes(activeSubTab)"><input type="checkbox" v-model="showContours"> 显示轮廓</label>
                        <label v-if="activeSubTab === 'pose'"><input type="checkbox" v-model="showPose"> 显示姿态</label>
                        <label v-if="activeSubTab === 'track'"><input type="checkbox" v-model="showTracks"> 显示历史轨迹 (仅视频流有效)</label>
                    </div>

                    <div class="control-group" v-if="activeSubTab !== 'pose'">
                        <label>类别筛选 (Classes)</label>
                        <div class="class-filters" style="max-height: 200px; overflow-y: auto; background: rgba(0,0,0,0.03); padding: 10px; border-radius: 8px;">
                            <div v-for="(name, id) in allNormalClasses" :key="id" style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
                                <input type="checkbox" :id="'cls-'+id" :value="id" v-model="selectedClasses">
                                <label :for="'cls-'+id" style="margin-bottom:0; cursor:pointer; color: var(--text-main);">{{ name }}</label>
                            </div>
                        </div>
                        <div style="margin-top:8px; display:flex; gap:10px;">
                            <button class="btn-secondary" style="padding:4px 8px; font-size:12px;" @click="selectedClasses = []">清空</button>
                            <button class="btn-secondary" style="padding:4px 8px; font-size:12px;" @click="selectedClasses = ['0']">只看人</button>
                        </div>
                    </div>
                </div>
            </div>

            <div class="main-panel">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                    <h3 style="margin:0;">3. 推理可视化</h3>
                    <div v-if="actualModelName" style="background:rgba(162, 28, 175, 0.1); color:var(--primary-accent); padding:4px 12px; border-radius:12px; font-size:13px; font-weight:600;">
                        运行中引擎: {{ actualModelName }}
                    </div>
                </div>
                <div v-if="!imageUrl" class="empty-state">请加载一张图片。</div>
                <div v-else>
                    <div class="result-display" style="position:relative; background:#eee; border-radius:12px; overflow:hidden; max-height:70vh; height:100%; display:flex; align-items:center; justify-content:center;">
                        <img :src="resultImageUrl || imageUrl" class="preview-img" style="width:100%; height:100%; object-fit: contain;">
                        <div v-if="isLoading" class="loading-overlay" style="position:absolute; top:0; left:0; right:0; bottom:0; background:rgba(255,255,255,0.7); display:flex; align-items:center; justify-content:center; flex-direction:column; gap:12px; z-index:10;">
                            <div class="spinner"></div>
                            <span style="font-weight:600; color:var(--primary-accent); max-width: 80%; text-align:center;">正在初始化或下载模型参数...<br>这可能需要数十秒至几分钟，请耐心静候</span>
                        </div>
                    </div>
                    
                    <div v-if="counts && Object.keys(counts).length > 0" class="stats-panel" style="margin-top:20px;">
                        <h4>检测统计</h4>
                        <div class="tags-container" style="display:flex; flex-wrap:wrap; gap:10px;">
                            <div v-for="(count, name) in counts" :key="name" class="tag" style="background:var(--panel-bg); color:var(--primary-accent); padding:6px 15px; border-radius:20px; font-size:14px; border:1px solid var(--primary-accent); display:flex; align-items:center; gap:8px; font-weight:600;">
                                <span class="tag-label">{{ name }}:</span>
                                <span class="tag-count">{{ count }}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    `,
    setup() {
        const imageUrl = ref('');
        const selectedFile = ref(null);
        const activeSubTab = ref('detect');

        const availableModels = computed(() => {
            if (activeSubTab.value === 'segment') {
                return [
                    { value: 'yolo26n-seg.pt', label: 'yolo26n-seg.pt' },
                    { value: 'yolo26s-seg.pt', label: 'yolo26s-seg.pt' },
                    { value: 'yolo26m-seg.pt', label: 'yolo26m-seg.pt' },
                    { value: 'yolo11n-seg.pt', label: 'yolo11n-seg.pt' },
                    { value: 'yolo11s-seg.pt', label: 'yolo11s-seg.pt' },
                    { value: 'yolov8n-seg.pt', label: 'yolov8n-seg.pt' },
                    { value: 'yolov8s-seg.pt', label: 'yolov8s-seg.pt' }
                ];
            } else if (activeSubTab.value === 'pose') {
                // Corrected user prompt mistake '-seg' -> '-pose' for pose estimation
                return [
                    { value: 'yolo26n-pose.pt', label: 'yolo26n-pose.pt' },
                    { value: 'yolo26s-pose.pt', label: 'yolo26s-pose.pt' },
                    { value: 'yolo26m-pose.pt', label: 'yolo26m-pose.pt' },
                    { value: 'yolo11n-pose.pt', label: 'yolo11n-pose.pt' },
                    { value: 'yolo11s-pose.pt', label: 'yolo11s-pose.pt' },
                    { value: 'yolov8n-pose.pt', label: 'yolov8n-pose.pt' },
                    { value: 'yolov8s-pose.pt', label: 'yolov8s-pose.pt' }
                ];
            } else if (activeSubTab.value === 'track') {
                return [
                    { value: 'yolo26n.pt', label: 'yolo26n.pt' },
                    { value: 'yolo26s.pt', label: 'yolo26s.pt' },
                    { value: 'yolov8s.pt', label: 'yolov8s.pt' },
                    { value: 'yolo26n-seg.pt', label: 'yolo26n-seg.pt' },
                    { value: 'yolo26s-seg.pt', label: 'yolo26s-seg.pt' }
                ];
            } else {
                return [
                    { value: 'yolo26n.pt', label: 'yolo26n.pt' },
                    { value: 'yolo26s.pt', label: 'yolo26s.pt' },
                    { value: 'yolo26m.pt', label: 'yolo26m.pt' },
                    { value: 'yolo11n.pt', label: 'yolo11n.pt' },
                    { value: 'yolo11s.pt', label: 'yolo11s.pt' },
                    { value: 'yolov8n.pt', label: 'yolov8n.pt' },
                    { value: 'yolov8s.pt', label: 'yolov8s.pt' }
                ];
            }
        });

        const modelName = ref('yolov8n.pt');
        const conf = ref(0.25);
        const iou = ref(0.45);
        const showLabels = ref(true);
        const showBoxes = ref(true);
        const showConfText = ref(true);
        const showMasks = ref(true);
        const showContours = ref(false);
        const showPose = ref(true);
        const actualModelName = ref('');

        const resultImageUrl = ref('');
        const counts = ref({});
        const isLoading = ref(false);

        const allNormalClasses = ref({});
        const selectedClasses = ref([]);
        const showTracks = ref(true);

        const ws = ref(null);
        const isStreaming = ref(false);

        const connectWS = () => {
            if (ws.value && ws.value.readyState === WebSocket.OPEN) return Promise.resolve();
            return new Promise((resolve, reject) => {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const host = window.location.hostname;
                const port = (window.location.port === '8080' || window.location.port === '') ? '8000' : window.location.port;
                const wsUrl = `${protocol}//${host}:${port}/api/yolo/ws/detect`;

                console.log("Connecting to YOLO WebSocket:", wsUrl);
                ws.value = new WebSocket(wsUrl);
                ws.value.onopen = () => { resolve(); };
                ws.value.onerror = (e) => {
                    console.error("YOLO WS Error:", e);
                    isStreaming.value = false;
                    reject(new Error("YOLO WebSocket 连接失败"));
                };
                ws.value.onclose = () => {
                    isStreaming.value = false;
                };
                ws.value.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    if (data.type === 'error') {
                        console.error("YOLO WebSocket 捕获到错误回复:", data.msg);
                        isStreaming.value = false;
                        return;
                    }
                    resultImageUrl.value = data.image_b64;
                    counts.value = data.counts;
                    actualModelName.value = data.actual_model_name || '';
                    isStreaming.value = false; // Reset lock
                };
            });
        };

        const onStreamFrame = async (b64) => {
            // Only stream if we aren't currently waiting for a frame
            if (isStreaming.value) return;

            // Unlock the empty state UI rendering barrier
            if (!imageUrl.value) {
                imageUrl.value = b64;
            }

            try {
                await connectWS();
                isStreaming.value = true;
                ws.value.send(JSON.stringify({
                    image: b64,
                    model_name: modelName.value,
                    conf: conf.value,
                    iou: iou.value,
                    classes: selectedClasses.value,
                    mode: activeSubTab.value,
                    show_tracks: showTracks.value,
                    show_boxes: showBoxes.value,
                    show_masks: showMasks.value,
                    show_contours: showContours.value,
                    show_labels: showLabels.value,
                    show_conf: showConfText.value,
                    show_pose: showPose.value
                }));
            } catch (e) {
                console.error(e);
            }
        };

        const fetchClasses = async () => {
            try {
                const res = await axios.get('/api/yolo/classes', { params: { model_name: modelName.value } });
                allNormalClasses.value = res.data;
            } catch (e) {
                console.error("Failed to fetch YOLO classes", e);
            }
        };

        onMounted(() => {
            fetchClasses();
        });

        watch(activeSubTab, (newTab) => {
            const valid = availableModels.value.map(m => m.value);
            if (!valid.includes(modelName.value)) {
                modelName.value = valid[0];
            }
        });

        watch(modelName, () => {
            fetchClasses();
        });

        watch([modelName, conf, iou, selectedClasses, activeSubTab, showLabels, showBoxes, showMasks, showConfText, showContours, showPose, showTracks], () => {
            // Static image reload
            if (imageUrl.value && !isStreaming.value && selectedFile.value) {
                requestDetection();
            }
            // Streaming reload (Force breaking WS to reconnect on next frame)
            if (isStreaming.value && ws.value && ws.value.readyState === WebSocket.OPEN) {
                ws.value.close();
                ws.value = null;
                isStreaming.value = false;
            }
        }, { deep: true });

        const onImageSelected = (payload) => {
            if (payload && payload.data) {
                selectedFile.value = payload.data;
                imageUrl.value = URL.createObjectURL(payload.data);
                resultImageUrl.value = '';
                counts.value = {};
                requestDetection();
            }
        };

        const requestDetection = async () => {
            if (!selectedFile.value) return;
            isLoading.value = true;

            const formData = new FormData();
            formData.append('file', selectedFile.value);
            formData.append('model_name', modelName.value);
            formData.append('conf', conf.value);
            formData.append('iou', iou.value);
            formData.append('classes', JSON.stringify(selectedClasses.value));
            formData.append('mode', activeSubTab.value);
            formData.append('show_boxes', showBoxes.value);
            formData.append('show_masks', showMasks.value);
            formData.append('show_contours', showContours.value);
            formData.append('show_labels', showLabels.value);
            formData.append('show_conf', showConfText.value);
            formData.append('show_pose', showPose.value);

            try {
                const res = await axios.post('/api/yolo/detect', formData);
                resultImageUrl.value = res.data.image_b64;
                counts.value = res.data.counts;
                actualModelName.value = res.data.actual_model_name || '';
            } catch (err) {
                console.error("YOLO Detection failed:", err);
                alert("检测失败: " + (err.response?.data?.detail || err.message));
            } finally {
                isLoading.value = false;
            }
        };

        return {
            imageUrl,
            availableModels,
            modelName,
            conf,
            iou,
            showLabels,
            showBoxes,
            showConfText,
            showMasks,
            showContours,
            showPose,
            actualModelName,
            resultImageUrl,
            counts,
            isLoading,
            onImageSelected,
            onStreamFrame,
            requestDetection,
            activeSubTab,
            allNormalClasses,
            selectedClasses,
            showTracks
        };
    }
}
