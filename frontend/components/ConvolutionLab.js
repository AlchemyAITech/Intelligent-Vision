import { ref, onMounted, onUnmounted, nextTick, computed, watch } from 'vue';
import ImageSource from './ImageSource.js';

export default {
    name: 'ConvolutionLab',
    components: {
        ImageSource
    },
    template: `
    <div class="conv-lab">
        <h2>⚙️ 卷积实验室</h2>
        <div class="layout-grid">
            
            <div class="sidebar-panel">
                <h3>1. 图像源</h3>
                <ImageSource @image-selected="onImageSelected" />
                <hr>
                
                <h3>2. 卷积设置</h3>
                <div v-if="imageUrl" class="settings">
                    <div class="control-group">
                        <label>处理模式 (输入)</label>
                        <select v-model="processMode" @change="requestConvolution">
                            <option value="RGB (彩色)">RGB (彩色)</option>
                            <option value="Grayscale (灰度)">Grayscale (灰度)</option>
                        </select>
                    </div>
                    
                    <div class="control-group">
                        <label>滤波器类别</label>
                        <select v-model="selectedCategory">
                            <option v-for="(cats, catName) in kernels" :key="catName" :value="catName">
                                {{ catName }}
                            </option>
                        </select>
                    </div>

                    <div class="control-group" v-if="availableKernelNames.length > 0">
                        <label>具体卷积核</label>
                        <select v-model="selectedKernelName" @change="requestConvolution">
                            <option v-for="kname in availableKernelNames" :key="kname" :value="kname">
                                {{ kname }}
                            </option>
                        </select>
                    </div>

                    <div class="kernel-matrix" v-if="currentKernelMatrix">
                        <h4>当前矩阵</h4>
                        <div class="matrix-container">
                            <table class="matrix-table">
                                <tr v-for="(row, rIdx) in currentKernelMatrix" :key="rIdx">
                                    <td v-for="(val, cIdx) in row" :key="cIdx">{{ Number(val).toFixed(2).replace(/\.00$/, '') }}</td>
                                </tr>
                            </table>
                        </div>
                    </div>

                    <div class="control-group checkbox">
                        <label>
                            <input type="checkbox" v-model="invertColor" @change="requestConvolution"> 反转颜色 (Invert Color)
                        </label>
                    </div>
                </div>
            </div>

            <div class="main-panel">
                <h3>3. 处理结果</h3>
                <div v-if="!imageUrl" class="empty-state">请加载一张图片。</div>
                <div v-else>
                    <div class="image-compare">
                        <div class="img-box">
                            <h4>输入图像</h4>
                            <img :src="imageUrl" class="preview-img">
                        </div>
                        <div class="img-box">
                            <h4>卷积结果</h4>
                            <div v-if="isLoading" class="loading-state">计算中...</div>
                            <img v-else-if="resultImageUrl" :src="resultImageUrl" class="preview-img">
                        </div>
                    </div>

                    <!-- New Dynamic Demo Section (Enhanced Visibility) -->
                    <div class="demo-section" style="margin-top:40px; border-top:2px solid var(--panel-border); padding-top:20px;">
                        <h3 style="color:var(--primary-accent); display:flex; align-items:center; gap:8px;">
                            <span>⚡ 4. 动态卷积演示 (实时扫描)</span>
                            <span v-if="isLoading" style="font-size:12px; font-weight:normal; color:var(--text-muted);">(正在预处理图像...)</span>
                        </h3>
                        
                        <div v-if="isLoading" class="empty-state" style="padding:40px 20px;">
                            <div class="loading-state">🚀 正在生成计算结果及动画切片，请稍候...</div>
                        </div>
                        
                        <div v-else-if="!resultImageUrl" class="empty-state" style="padding:40px 20px; background:rgba(162, 28, 175, 0.02); border-radius:12px; border:1px dashed var(--panel-border);">
                            <p style="color:var(--primary-accent); font-weight:600;">请先从左侧面板选择“具体卷积核”以激活此演示模块。</p>
                        </div>
                        
                        <div v-else>
                            <div class="demo-controls" style="display:flex; align-items:center; gap:20px; margin-bottom:15px; background:rgba(162, 28, 175, 0.05); padding:15px; border-radius:12px;">
                                <button v-if="!isDemoRunning" class="btn-primary" @click="startDemo" style="min-width:120px;">▶ 开始扫描</button>
                                <button v-else class="btn-danger" @click="stopDemo" style="min-width:120px;">⏹ 停止扫描</button>
                                <button class="btn-secondary" @click="resetDemo">🔄 重置</button>
                                
                                <div style="flex:1; display:flex; align-items:center; gap:10px;">
                                    <label style="font-size:12px; color:var(--text-muted); white-space:nowrap;">扫描速度:</label>
                                    <input type="range" v-model="demoSpeed" min="5" max="200" step="5" style="flex:1;">
                                    <span style="font-size:12px; min-width:35px; text-align:right;">{{ 205 - demoSpeed }}ms</span>
                                </div>
                            </div>

                            <div class="demo-canvases" style="display:flex; gap:20px; justify-content:center; background:#000; padding:20px; border-radius:12px; min-height:220px; border:1px solid #333;">
                                <div style="text-align:center;">
                                    <div style="color:#aaa; font-size:11px; margin-bottom:8px; letter-spacing:1px;">输入图像 (卷积核扫描中)</div>
                                    <canvas ref="inputCanvas" style="max-width:300px; border:1px solid #444; height:auto; background:#111;"></canvas>
                                </div>
                                <div style="display:flex; align-items:center; color:var(--primary-accent); font-size:24px; opacity:0.6;">➡</div>
                                <div style="text-align:center;">
                                    <div style="color:#aaa; font-size:11px; margin-bottom:8px; letter-spacing:1px;">卷积结果 (逐像素揭示)</div>
                                    <canvas ref="outputCanvas" style="max-width:300px; border:1px solid #444; height:auto; background:#111;"></canvas>
                                </div>
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
        const resultImageUrl = ref('');
        const isLoading = ref(false);

        const kernels = ref({});
        const selectedCategory = ref('');
        const selectedKernelName = ref('');
        const processMode = ref('RGB (彩色)');
        const invertColor = ref(false);

        const isDemoRunning = ref(false);
        const demoSpeed = ref(100);
        const inputCanvas = ref(null);
        const outputCanvas = ref(null);
        let scanInterval = null;
        let currentScanX = 0;
        let currentScanY = 0;

        // --- Logic ---
        const loadKernels = async () => {
            try {
                const res = await axios.get('/api/image/kernels');
                kernels.value = res.data;
                const cats = Object.keys(kernels.value);
                if (cats.length > 0) {
                    selectedCategory.value = cats[0];
                    const knames = Object.keys(kernels.value[selectedCategory.value]);
                    if (knames.length > 0) {
                        selectedKernelName.value = knames[0];
                    }
                }
            } catch (e) {
                console.error("Failed to load kernels", e);
            }
        };

        onMounted(() => {
            loadKernels();
        });

        onUnmounted(() => {
            stopDemo();
        });

        // Watch category change to update kernel names
        watch(selectedCategory, (newCat) => {
            if (kernels.value[newCat]) {
                const knames = Object.keys(kernels.value[newCat]);
                if (knames.length > 0 && !knames.includes(selectedKernelName.value)) {
                    selectedKernelName.value = knames[0];
                    requestConvolution();
                }
            }
        });

        const availableKernelNames = computed(() => {
            if (!selectedCategory.value || !kernels.value[selectedCategory.value]) return [];
            return Object.keys(kernels.value[selectedCategory.value]);
        });

        const currentKernelMatrix = computed(() => {
            if (!selectedCategory.value || !selectedKernelName.value) return null;
            return kernels.value[selectedCategory.value]?.[selectedKernelName.value];
        });

        const formatMatrix = (matrix) => {
            if (!matrix) return "";
            return matrix.map(row =>
                "[" + row.map(v => v.toFixed(3).padStart(6, ' ')).join(', ') + "]"
            ).join("\\n");
        };

        const onImageSelected = (payload) => {
            if (payload && payload.data) {
                selectedFile.value = payload.data;
                imageUrl.value = URL.createObjectURL(payload.data);
                requestConvolution();
            }
        };

        const requestConvolution = async () => {
            if (!selectedFile.value || !selectedCategory.value || !selectedKernelName.value) return;

            isLoading.value = true;
            stopDemo();
            const formData = new FormData();
            formData.append('file', selectedFile.value);
            formData.append('process_mode', processMode.value);
            formData.append('category', selectedCategory.value);
            formData.append('kernel_name', selectedKernelName.value);
            formData.append('invert_color', invertColor.value);

            try {
                const res = await axios.post('/api/image/convolution', formData);
                resultImageUrl.value = res.data.image_b64;
                nextTick(() => { resetDemo(); });
            } catch (err) {
                console.error("Convolution failed:", err);
                alert("卷积处理失败: " + (err.response?.data?.detail || err.message));
            } finally {
                isLoading.value = false;
            }
        };

        const resetDemo = () => {
            stopDemo();
            currentScanX = 0;
            currentScanY = 0;
            if (!inputCanvas.value || !outputCanvas.value || !imageUrl.value || !resultImageUrl.value) return;

            const imgIn = new Image();
            imgIn.onload = () => {
                inputCanvas.value.width = imgIn.width;
                inputCanvas.value.height = imgIn.height;
                const ctx = inputCanvas.value.getContext('2d');
                ctx.drawImage(imgIn, 0, 0);

                // Prepare output canvas
                outputCanvas.value.width = imgIn.width;
                outputCanvas.value.height = imgIn.height;
                const ctxOut = outputCanvas.value.getContext('2d');
                ctxOut.fillStyle = '#111';
                ctxOut.fillRect(0, 0, imgIn.width, imgIn.height);
            };
            imgIn.src = imageUrl.value;
        };

        const startDemo = () => {
            if (isDemoRunning.value) return;
            isDemoRunning.value = true;

            const imgIn = new Image();
            imgIn.src = imageUrl.value;
            const imgOut = new Image();
            imgOut.src = resultImageUrl.value;

            const runStep = () => {
                if (!isDemoRunning.value) return;

                const ctxIn = inputCanvas.value.getContext('2d');
                const ctxOut = outputCanvas.value.getContext('2d');
                const W = inputCanvas.value.width;
                const H = inputCanvas.value.height;

                // Step 1: Draw Input + Kernel Marker
                ctxIn.clearRect(0, 0, W, H);
                ctxIn.drawImage(imgIn, 0, 0);

                // Draw kernel rect (centered at scan pos)
                ctxIn.strokeStyle = '#ff4757';
                ctxIn.lineWidth = 2;
                ctxIn.strokeRect(currentScanX - 1, currentScanY - 1, 3, 3);

                // Add soft overlay
                ctxIn.fillStyle = 'rgba(255, 71, 87, 0.2)';
                ctxIn.fillRect(currentScanX - 1, currentScanY - 1, 3, 3);

                // Step 2: Draw Output result pixel
                // We use drawImage with source rect to copy from processed image
                // To make it look "drawing", we copy a 1x1 block
                ctxOut.drawImage(imgOut, currentScanX, currentScanY, 1, 1, currentScanX, currentScanY, 1, 1);

                // Step 3: Fast Multi-Step Movement
                // Use quadratic scaling so high speeds are drastically faster
                // At max speed (200), steps = 40000 / 8 = 5000 pixels per frame
                const stepsPerFrame = Math.max(1, Math.floor((demoSpeed.value * demoSpeed.value) / 8));

                for (let s = 0; s < stepsPerFrame; s++) {
                    ctxOut.drawImage(imgOut, currentScanX, currentScanY, 1, 1, currentScanX, currentScanY, 1, 1);

                    currentScanX += 1;
                    if (currentScanX >= W) {
                        currentScanX = 0;
                        currentScanY += 1;
                    }
                    if (currentScanY >= H) break;
                }

                if (currentScanY >= H) {
                    stopDemo();
                    return;
                }

                // Adjust delay: 0 at high speed to rely purely on browser rendering rate
                const delay = Math.max(0, 50 - demoSpeed.value / 2);
                scanInterval = setTimeout(runStep, delay);
            };

            runStep();
        };

        const stopDemo = () => {
            isDemoRunning.value = false;
            if (scanInterval) clearTimeout(scanInterval);
            scanInterval = null;
        };

        return {
            imageUrl,
            processMode,
            kernels,
            selectedCategory,
            selectedKernelName,
            availableKernelNames,
            currentKernelMatrix,
            invertColor,
            resultImageUrl,
            onImageSelected,
            requestConvolution,
            formatMatrix,
            isDemoRunning,
            demoSpeed,
            inputCanvas,
            outputCanvas,
            startDemo,
            stopDemo,
            resetDemo
        };
    }
}
