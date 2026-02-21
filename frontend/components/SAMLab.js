import { ref, onMounted, onUnmounted, nextTick, computed } from 'vue';
import ImageSource from './ImageSource.js';

export default {
    name: 'SAMLab',
    components: {
        ImageSource
    },
    template: `
    <div class="sam-lab-unified">
        <div class="lab-header" style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
            <h2 style="margin:0;">✨ SAM 3 万物分割实验室</h2>
            <div class="tabs" style="margin-bottom:0; border-bottom:none;">
                <button :class="{active: subTab === 'labeling'}" @click="subTab = 'labeling'">交互式标注</button>
                <button :class="{active: subTab === 'tracking'}" @click="subTab = 'tracking'">零样本追踪</button>
                <button :class="{active: subTab === 'recognition'}" @click="subTab = 'recognition'">零样本识别</button>
            </div>
        </div>

        <!-- Using the standard layout-grid from index.html -->
        <div class="layout-grid" style="display: grid; grid-template-columns: 320px 1fr; gap: 20px;">
            <!-- Sidebar Panel -->
            <div class="sidebar-panel">
                <div v-if="subTab === 'labeling'">
                    <h3>1. 静态图像源</h3>
                    <div class="control-group">
                        <ImageSource @image-selected="handleFileUpload" :hideCaptureBtn="true" />
                    </div>
                    <hr>
                    <h3>2. 标注工具</h3>
                    <div class="control-group" style="display:flex; flex-direction:column; gap:12px;">
                        <button class="btn-danger" style="width:100%;" @click="resetPrompts" :disabled="!sessionId || isLoading">
                            🔄 清除所有标注
                        </button>
                        <div class="instructions" style="font-size:12px; color:var(--text-muted); padding:10px; background:rgba(0,0,0,0.03); border-radius:8px;">
                            <p style="margin:0 0 5px 0;">🔴 负向提示点 (右键)</p>
                            <p style="margin:0;">🟢 正向提示点 (左键)</p>
                        </div>
                    </div>
                </div>

                <div v-if="subTab === 'tracking' || subTab === 'recognition'">
                    <h3>1. 实时流媒体源</h3>
                    <div class="control-group">
                        <ImageSource @stream-frame="onStreamFrame" :hideVideoPreview="false" />
                    </div>
                    <hr>
                    <h3>2. 任务控制</h3>
                    <div class="control-group" style="display:flex; flex-direction:column; gap:12px;">
                        <button v-if="subTab === 'tracking'" class="btn-primary" @click="toggleTracking" :disabled="!sessionId">
                            {{ isTracking ? '⏹ 停止动态追踪' : '▶ 开启实时追踪' }}
                        </button>
                        <button v-if="subTab === 'recognition'" class="btn-primary" @click="triggerIdentify" :disabled="!sessionId || isLoading">
                            🔍 识别选定对象
                        </button>
                        <button class="btn-secondary" @click="resetPrompts">🔄 重置状态</button>
                    </div>
                </div>

                <div v-if="sessionId" class="status-panel" style="margin-top:20px;">
                    <div :class="['status-bar', { loading: isLoading }]" style="font-size:12px; margin:0;">
                         {{ statusMessage }}
                    </div>
                </div>
            </div>

            <!-- Main Panel -->
            <div class="main-panel" style="display:flex; flex-direction:column;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
                    <h3 style="margin:0; border:none;">3. 推理可视化工作区</h3>
                    <div v-if="sessionId" style="font-size:12px; color:var(--primary-accent); font-weight:700;">
                        Session ID: {{ sessionId.slice(0,8) }}...
                    </div>
                </div>

                <div class="workspace" style="background:#0a0a0a; border-radius:12px; overflow:hidden; position:relative; flex-grow:1; display:flex; align-items:center; justify-content:center; border: 1px solid var(--panel-border); box-shadow: inset 0 0 20px rgba(0,0,0,0.5);">
                    <div class="canvas-container" style="position:relative; cursor: crosshair;" @contextmenu.prevent>
                        <img ref="imageRef" :src="imageUrl" v-if="imageUrl" class="img-preview" 
                             style="max-width:100%; max-height:70vh; display:block; border-radius:4px;" @load="initCanvases">
                        
                        <canvas ref="maskCanvas" style="position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none; z-index:10; opacity:0.8;"></canvas>
                        <canvas ref="eventCanvas" style="position:absolute; top:0; left:0; width:100%; height:100%; z-index:20;"
                            @mousedown="handleCanvasClick">
                        </canvas>

                        <div v-if="subTab === 'recognition' && recognitionResult" 
                             style="position:absolute; top:20px; right:20px; z-index:100; background:rgba(162, 28, 175, 0.9); color:white; padding:10px 20px; border-radius:12px; font-weight:800; border:1px solid rgba(255,255,255,0.2); box-shadow:0 4px 15px rgba(0,0,0,0.3); backdrop-filter:blur(10px); animation: fadeInDown 0.3s;">
                            🏷️ 识别为: {{ recognitionResult }}
                        </div>
                    </div>

                    <div v-if="!imageUrl" class="empty-state" style="border:none; background:transparent;">
                        <div style="font-size:48px; margin-bottom:15px; opacity:0.3;">🔭</div>
                        等待图像输入以开启万物分割实验
                    </div>
                </div>
            </div>
        </div>
    </div>
    `,

    setup() {
        const API_BASE = window.location.origin + "/api/sam";

        const subTab = ref('labeling');
        const imageRef = ref(null);
        const maskCanvas = ref(null);
        const eventCanvas = ref(null);

        const imageUrl = ref('');
        const sessionId = ref('');
        const statusMessage = ref('等待图像输入...');
        const isLoading = ref(false);

        const points = ref([]);
        const recognitionResult = ref('');
        const isTracking = ref(false);

        let maskImgElement = new Image();

        // 统一处理文件/帧上传
        const handleFileUpload = async (payload) => {
            const file = payload.data;
            if (!file) return;

            imageUrl.value = URL.createObjectURL(file);
            points.value = [];
            clearCanvases();
            recognitionResult.value = '';

            isLoading.value = true;
            statusMessage.value = '特征提取中 (Feature Embedding)...';

            const formData = new FormData();
            formData.append('file', file);

            try {
                const res = await axios.post(`${API_BASE}/upload`, formData);
                sessionId.value = res.data.session_id;
                statusMessage.value = '模型就绪，请在图像上点击目标。';
            } catch (error) {
                statusMessage.value = '上传失败';
            } finally {
                isLoading.value = false;
                await nextTick();
                initCanvases();
            }
        };

        const onStreamFrame = async (b64) => {
            // 在标注模式下不自动响应流，除非手动抓拍
            if (subTab.value === 'labeling') return;

            // 如果处于追踪模式，每帧发送请求
            if (isTracking.value && sessionId.value) {
                // TODO: 追踪逻辑实现
            }

            // 初始化背景
            if (!imageUrl.value) {
                imageUrl.value = b64;
                // 自动执行一次静默上传以获取特征
                fetch(b64).then(r => r.blob()).then(blob => {
                    handleFileUpload({ data: blob });
                });
            }
        };

        const initCanvases = () => {
            const img = imageRef.value;
            if (!img) return;

            const mCanvas = maskCanvas.value;
            const eCanvas = eventCanvas.value;
            if (!mCanvas || !eCanvas) return;

            // 同步画布尺寸与图片显示尺寸
            mCanvas.width = img.clientWidth;
            mCanvas.height = img.clientHeight;
            eCanvas.width = img.clientWidth;
            eCanvas.height = img.clientHeight;

            if (maskImgElement.src) drawMaskImage();
            redrawPoints();
        };

        const handleCanvasClick = async (event) => {
            if (!sessionId.value || isLoading.value) return;

            const rect = eventCanvas.value.getBoundingClientRect();
            const dispX = event.clientX - rect.left;
            const dispY = event.clientY - rect.top;

            const img = imageRef.value;
            const scaleX = img.naturalWidth / img.clientWidth;
            const scaleY = img.naturalHeight / img.clientHeight;

            const realX = dispX * scaleX;
            const realY = dispY * scaleY;
            const label = event.button === 2 ? 0 : 1;

            points.value.push({ x: realX, y: realY, label });
            redrawPoints();

            await requestPrediction();
        };

        const redrawPoints = () => {
            if (!eventCanvas.value) return;
            const ctx = eventCanvas.value.getContext('2d');
            ctx.clearRect(0, 0, eventCanvas.value.width, eventCanvas.value.height);

            const img = imageRef.value;
            if (!img) return;

            const scaleX = img.naturalWidth / img.clientWidth;
            const scaleY = img.naturalHeight / img.clientHeight;

            points.value.forEach(pt => {
                const dx = pt.x / scaleX;
                const dy = pt.y / scaleY;
                ctx.beginPath();
                ctx.arc(dx, dy, 6, 0, 2 * Math.PI);
                ctx.fillStyle = pt.label === 1 ? '#2ecc71' : '#e74c3c';
                ctx.fill();
                ctx.lineWidth = 2;
                ctx.strokeStyle = '#FFFFFF';
                ctx.stroke();
            });
        };

        const requestPrediction = async () => {
            isLoading.value = true;
            statusMessage.value = '逐像素分割中 (Segmenting)...';
            try {
                const payload = {
                    session_id: sessionId.value,
                    points: points.value,
                    boxes: []
                };
                const res = await axios.post(`${API_BASE}/predict`, payload);
                maskImgElement.onload = () => {
                    drawMaskImage();
                    statusMessage.value = '子目标分割完成。';
                    isLoading.value = false;
                };
                maskImgElement.src = res.data.mask_base64;
            } catch (e) {
                statusMessage.value = '分割异常';
                isLoading.value = false;
            }
        };

        const drawMaskImage = () => {
            if (!maskCanvas.value || !maskImgElement.src) return;
            const ctx = maskCanvas.value.getContext('2d');
            ctx.clearRect(0, 0, maskCanvas.value.width, maskCanvas.value.height);
            ctx.drawImage(maskImgElement, 0, 0, maskCanvas.value.width, maskCanvas.value.height);
        };

        const triggerIdentify = async () => {
            if (!sessionId.value || points.value.length === 0) {
                alert("请先在画面中通过点选指定待识别物体");
                return;
            }
            isLoading.value = true;
            statusMessage.value = '正在提取语义语义指纹 (Identifying)...';
            try {
                // 识别通常只需要预测出的 Mask 所对应的特征
                const res = await axios.post(`${API_BASE}/identify`, {
                    session_id: sessionId.value,
                    points: points.value
                });
                recognitionResult.value = res.data.label || '未知物体';
                statusMessage.value = '识别完成。';
            } catch (e) {
                statusMessage.value = '识别请求失败';
            } finally {
                isLoading.value = false;
            }
        };

        const toggleTracking = () => {
            isTracking.value = !isTracking.value;
            if (isTracking.value) {
                statusMessage.value = '追踪引擎激活，正在解算运动矢量...';
            } else {
                statusMessage.value = '追踪已停止。';
            }
        };

        const resetPrompts = () => {
            points.value = [];
            maskImgElement = new Image();
            recognitionResult.value = '';
            isTracking.value = false;
            clearCanvases();
            statusMessage.value = '工作区已清空。';
        };

        const clearCanvases = () => {
            [maskCanvas.value, eventCanvas.value].forEach(c => {
                if (c) c.getContext('2d').clearRect(0, 0, c.width, c.height);
            });
        };

        onMounted(() => {
            window.addEventListener('resize', initCanvases);
        });

        onUnmounted(() => {
            window.removeEventListener('resize', initCanvases);
            if (imageUrl.value && imageUrl.value.startsWith('blob:')) {
                URL.revokeObjectURL(imageUrl.value);
            }
        });

        return {
            subTab, imageRef, maskCanvas, eventCanvas,
            imageUrl, sessionId, statusMessage, isLoading,
            points, recognitionResult, isTracking,
            handleFileUpload, onStreamFrame, handleCanvasClick,
            resetPrompts, initCanvases, toggleTracking, triggerIdentify
        };
    }
}
