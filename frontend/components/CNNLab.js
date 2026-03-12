import { ref, onMounted, onUnmounted, nextTick, computed, watch } from 'vue';

export default {
    name: 'CNNLab',
    template: `
    <div class="cnn-lab">
        <div class="lab-header" style="margin-bottom:20px;">
            <h2>🧠 神经网络实验室 (CNN)</h2>
        </div>

        <div class="layout-grid" style="display: grid !important; grid-template-columns: 320px 1fr !important; gap: 20px; align-items: start;">
            <!-- LEFT COLUMN: Params -->
            <div class="sidebar-panel" style="position: sticky; top: 0;">
                <div class="tabs" style="display:flex; gap:10px; margin-bottom:20px; border-bottom:1px solid var(--panel-border); padding-bottom:10px;">
                    <button :class="{active: activeTab === 'train'}" @click="switchTab('train')" style="flex:1; font-size:13px; padding:8px 5px;">训练模块</button>
                    <button :class="{active: activeTab === 'test'}" @click="switchTab('test')" style="flex:1; font-size:13px; padding:8px 5px;">测试模块</button>
                </div>

                <!-- 1. Training Params -->
                <div v-if="activeTab === 'train'">
                    <h3>1. 训练参数</h3>
                    <div class="control-group">
                        <label>学习率 (Learning Rate)</label>
                        <select v-model="config.lr">
                            <option value="0.001">0.001</option>
                            <option value="0.01">0.01</option>
                            <option value="0.1">0.1</option>
                        </select>
                    </div>
                    <div class="control-group">
                        <label>训练轮数 (Epochs): {{ config.epochs }}</label>
                        <input type="range" min="1" max="10" v-model="config.epochs">
                    </div>
                    <div class="control-group">
                        <label>批大小 (Batch Size)</label>
                        <select v-model="config.batch_size">
                            <option value="32">32</option>
                            <option value="64">64</option>
                            <option value="128">128</option>
                        </select>
                    </div>
                    <div class="control-group">
                        <label>优化器 (Optimizer)</label>
                        <select v-model="config.optimizer">
                            <option value="Adam">Adam (自适应)</option>
                            <option value="SGD">SGD (随机梯度下降)</option>
                        </select>
                    </div>
                    <div class="control-group">
                        <label>损失函数 (Loss Function)</label>
                        <select v-model="config.loss_fn">
                            <option value="CrossEntropy">交叉熵 (CrossEntropy)</option>
                            <option value="NLLLoss">负对数似然 (NLLLoss)</option>
                        </select>
                    </div>
                    <div class="control-group">
                        <label>网络结构 (卷积通道数)</label>
                        <div style="display:flex; flex-direction:column; gap:8px;">
                            <div v-for="(ch, idx) in config.layers" :key="idx" style="display:flex; align-items:center; gap:10px; background:rgba(162, 28, 175, 0.05); padding:8px; border-radius:8px; border:1px solid var(--panel-border);">
                                <span style="font-size:12px; color:var(--text-muted); width:40px;">L{{idx + 1}}</span>
                                <input type="number" v-model.number="config.layers[idx]" min="1" max="128" style="padding:4px; font-size:13px; width:70px;">
                                <button class="btn-secondary" style="padding:4px 8px; font-size:12px; border:none; background:transparent;" @click="removeLayer(idx)" :disabled="config.layers.length <= 1">✕</button>
                            </div>
                            <button class="btn-secondary" style="font-size:12px; padding:6px; margin-top:4px;" @click="addLayer">+ 新增卷积层</button>
                        </div>
                    </div>
                    
                    <div class="control-group checkbox" style="margin-top:15px;">
                        <label>
                            <input type="checkbox" v-model="config.show_visuals"> 
                            <span style="font-size:14px; font-weight:600;">特征图显示</span>
                        </label>
                    </div>

                    <hr>
                    <div class="btn-group" style="display:flex; flex-direction:column; gap:10px;">
                        <!-- Button State Machine -->
                        <div v-if="!isTraining">
                            <button class="btn-primary" @click="startTraining" :disabled="!hasTorch" style="width:100%">🚀 开始训练</button>
                        </div>
                        <div v-else style="display:flex; gap:10px;">
                            <button v-if="!isPaused" class="btn-secondary" @click="pauseTraining" style="flex:1">⏸ 暂停</button>
                            <button v-else class="btn-primary" @click="resumeTraining" style="flex:1">▶️ 继续</button>
                            <button class="btn-danger" @click="stopTraining" style="flex:1">⏹ 停止</button>
                        </div>
                    </div>
                    
                    <div v-if="!hasTorch" style="color:red; margin-top:20px; font-size:12px;">
                        ⚠️ 服务端未检测到 PyTorch 模块。
                    </div>
                </div>

                <!-- 1. Test Input Sidebar -->
                <div v-else-if="activeTab === 'test'">
                    <h3>1. 测试输入</h3>
                    <div class="control-group">
                        <label>数字化手写板</label>
                        <div style="background:white; border:2px solid var(--panel-border); border-radius:8px; overflow:hidden; position:relative; width: 200px; height: 200px; margin: 0 auto; box-shadow: inset 0 0 10px rgba(0,0,0,0.1);">
                            <canvas ref="drawCanvas" width="200" height="200" 
                                @mousedown="startDrawing" @mousemove="draw" @mouseup="stopDrawing" @mouseleave="stopDrawing"
                                @touchstart.prevent="startDrawing" @touchmove.prevent="draw" @touchend.prevent="stopDrawing"
                                style="cursor:crosshair;"></canvas>
                        </div>
                        <div style="display:flex; gap:10px; margin-top:10px;">
                            <button class="btn-secondary" style="flex:1; font-size:12px;" @click="clearCanvas">擦除</button>
                            <button class="btn-primary" style="flex:1; font-size:12px;" @click="testCanvas">识别</button>
                        </div>
                    </div>
                    <hr>
                    <div class="control-group">
                        <label>其他测试方式</label>
                        <div style="display:flex; gap:10px;">
                            <button class="btn-secondary" style="flex:1; font-size:12px;" @click="testRandom">🎲 随机样本</button>
                            <button class="btn-secondary" style="flex:1; font-size:12px;" @click="$refs.fileInput.click()">📁 上传文件</button>
                            <input type="file" ref="fileInput" @change="uploadTestImg" accept="image/*" style="display:none;">
                        </div>
                    </div>
                </div>
            </div>

            <!-- RIGHT COLUMN: Status -->
            <div class="main-panel">
                <!-- 2. Training Status -->
                <div v-if="activeTab === 'train'">
                    <div class="training-header" style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px; border-bottom: 2px solid var(--panel-border); padding-bottom:10px;">
                        <h3 style="border:none; margin:0;">2. 训练状态</h3>
                        <div v-if="trainAcc > 0" style="display:flex; gap:15px;">
                            <div class="stat-badge" style="background:var(--primary-accent); color:white; padding:4px 12px; border-radius:15px; font-size:13px; font-weight:700;">
                                训练准确率: {{ trainAcc.toFixed(1) }}%
                            </div>
                            <div class="stat-badge" style="background:var(--secondary-accent); color:white; padding:4px 12px; border-radius:15px; font-size:13px; font-weight:700;">
                                测试准确率: {{ testAcc.toFixed(1) }}%
                            </div>
                        </div>
                    </div>
                    
                    <div class="status-bar" :class="{loading: isTraining}" style="margin-bottom:20px;">
                        {{ statusMsg }}
                    </div>
                    
                    <div class="section-title" style="font-size:14px; font-weight:600; color:var(--text-muted); margin-bottom:10px;">损失函数曲线 (Loss History)</div>
                    <div class="chart-container" style="width: 100%; aspect-ratio: 2.5 / 1; height: auto; background:#fff; border-radius:10px; border: 1px solid var(--panel-border); margin-bottom: 25px; position:relative; padding:10px; overflow: hidden; display: flex; align-items: center; justify-content: center;">
                        <!-- Using 1:2.5 aspect ratio viewBox 625x250 -->
                        <svg viewBox="0 0 625 250" style="width:100%; height:100%; font-family: 'Inter', sans-serif;">
                            <!-- Horizontal Grid Lines & Y Labels (Range 0-2.0, Step 0.2) -->
                            <!-- yZero=220, yRange=200 (Loss 2.0 at y=20) -> 20px per 0.2 Loss -->
                            <g stroke="#f8fafc" stroke-width="0.8">
                                <line x1="45" y1="220" x2="605" y2="220" /> <!-- 0.0 -->
                                <line x1="45" y1="200" x2="605" y2="200" /> <!-- 0.2 -->
                                <line x1="45" y1="180" x2="605" y2="180" /> <!-- 0.4 -->
                                <line x1="45" y1="160" x2="605" y2="160" /> <!-- 0.6 -->
                                <line x1="45" y1="140" x2="605" y2="140" /> <!-- 0.8 -->
                                <line x1="45" y1="120" x2="605" y2="120" /> <!-- 1.0 -->
                                <line x1="45" y1="100" x2="605" y2="100" /> <!-- 1.2 -->
                                <line x1="45" y1="80" x2="605" y2="80" />   <!-- 1.4 -->
                                <line x1="45" y1="60" x2="605" y2="60" />   <!-- 1.6 -->
                                <line x1="45" y1="40" x2="605" y2="40" />   <!-- 1.8 -->
                                <line x1="45" y1="20" x2="605" y2="20" />   <!-- 2.0 -->
                            </g>

                            <!-- Axis Labels (Y) - Micro Font 7.5 -->
                            <g font-size="7.5" fill="#94a3b8" text-anchor="end">
                                <text x="35" y="223">0.0</text>
                                <text x="35" y="203">0.2</text>
                                <text x="35" y="183">0.4</text>
                                <text x="35" y="163">0.6</text>
                                <text x="35" y="143">0.8</text>
                                <text x="35" y="123">1.0</text>
                                <text x="35" y="103">1.2</text>
                                <text x="35" y="83">1.4</text>
                                <text x="35" y="63">1.6</text>
                                <text x="35" y="43">1.8</text>
                                <text x="35" y="23">2.0</text>
                            </g>

                            <!-- Main Axes -->
                            <line x1="45" y1="220" x2="605" y2="220" stroke="#cbd5e1" stroke-width="1" /> <!-- X -->
                            <line x1="45" y1="220" x2="45" y2="20" stroke="#e2e8f0" stroke-width="0.8" /> <!-- Y -->

                            <!-- X Axis Labels -->
                            <g font-size="7.5" fill="#94a3b8" text-anchor="middle">
                                <text x="45" y="238">Step 0</text>
                                <text v-if="lossHistory.length > 0" x="605" y="238">Step {{ lossHistory.length * 10 }}</text>
                            </g>

                            <!-- Data Polyline (Thinner 1.2px) -->
                            <polyline :points="lossPath" fill="none" stroke="#4a69bd" stroke-width="1.2" stroke-linejoin="round" stroke-linecap="round" style="transition: all 0.3s;" />
                        </svg>
                    </div>

                    <!-- Bottom row: Feature Maps & Probs side-by-side -->
                    <div v-if="config.show_visuals" class="visuals-grid" style="display:grid; grid-template-columns: 1fr 1fr; gap:20px;">
                        <div class="feat-maps-container" style="background:rgba(255,255,255,0.5); padding:15px; border-radius:12px; border:1px solid var(--panel-border);">
                            <h4 style="margin-top:0;">动态特征图</h4>
                            <div v-if="featImg" style="text-align:center;">
                                <div style="font-size:11px; color:var(--text-muted); margin-bottom:8px;">输入标签: {{ trueLabel }} (200步更新图, 10步更新层)</div>
                                <img :src="featImg" style="width:100%; image-rendering:pixelated; border-radius:4px; border:1px solid #eee;">
                            </div>
                            <div v-else class="empty-state" style="height:150px; display:flex; align-items:center; justify-content:center; border:none; background:transparent;">等待数据...</div>
                        </div>

                        <div class="probs-viz-container" style="background:rgba(255,255,255,0.5); padding:15px; border-radius:12px; border:1px solid var(--panel-border);">
                            <h4 style="margin-top:0;">实时预测概率</h4>
                            <div v-if="probs.length" class="prob-chart">
                                <div v-for="(p, i) in probs" :key="i" class="prob-row" style="margin-bottom:6px;">
                                    <span style="width:20px; font-size:12px; font-weight:700;">{{i}}</span>
                                    <div style="flex:1; height:10px; background:rgba(0,0,0,0.05); border-radius:5px; overflow:hidden;">
                                        <div :style="{width: (p*100) + '%'}" style="height:100%; background:var(--primary-accent); transition: width 0.3s; box-shadow: 0 0 5px var(--primary-accent);"></div>
                                    </div>
                                    <span style="width:40px; font-size:11px; text-align:right;">{{ (p*100).toFixed(0) }}%</span>
                                </div>
                            </div>
                            <div v-else class="empty-state" style="height:150px; display:flex; align-items:center; justify-content:center; border:none; background:transparent;">等待推理...</div>
                        </div>
                    </div>
                    <div v-else class="visuals-disabled-msg" style="text-align:center; padding:40px; color:var(--text-muted); border:1px dashed var(--panel-border); border-radius:12px;">
                        💡 勾选左侧“特征图显示”可在训练时实时观察网络内部变化。
                    </div>
                </div>

                <!-- 2. Test Analysis Right Panel -->
                <div v-else-if="activeTab === 'test'">
                    <h3>2. 识别结果分析</h3>
                    <div v-if="!testImg" class="empty-state" style="padding:100px 20px;">请在展示区左侧进行识别输入。</div>
                    <div v-else style="display:flex; flex-direction:column; gap:20px;">
                        <div style="display:flex; gap:40px; align-items:flex-start;">
                            <div class="input-preview-card" style="text-align:center; background:white; padding:15px; border-radius:16px; border:1px solid var(--panel-border); box-shadow:0 4px 15px rgba(0,0,0,0.05);">
                                <img :src="testImg" style="width:220px; height:220px; image-rendering:pixelated; border-radius:8px;">
                                <div v-if="testTrueLabel !== null" style="margin-top:15px; font-size:18px;">
                                    真值标签: <span style="font-weight:800; color:var(--text-muted);">{{testTrueLabel}}</span>
                                </div>
                            </div>
                            
                            <div class="result-details" style="flex:1;">
                                <div v-if="testPrediction !== null">
                                    <div style="font-size:20px; margin-bottom:25px; background:rgba(162, 28, 175, 0.05); padding:15px; border-radius:12px; border-left:5px solid var(--primary-accent);">
                                        AI 识别为: <span style="font-size:48px; font-weight:900; color:var(--primary-accent); vertical-align:middle; margin-left:15px;">{{ testPrediction }}</span>
                                    </div>
                                    <h4>预测概率置信度</h4>
                                    <div class="prob-chart">
                                        <div v-for="(p, i) in testProbs" :key="i" class="prob-row" style="margin-bottom:8px;">
                                            <span style="width:25px; font-weight:700;">{{i}}</span>
                                            <div style="flex:1; height:16px; background:rgba(0,0,0,0.05); border-radius:8px; overflow:hidden;">
                                                <div :style="{width: (p*100) + '%'}" style="height:100%; background:linear-gradient(90deg, var(--primary-accent), var(--secondary-accent)); transition: width 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275);"></div>
                                            </div>
                                            <span style="width:50px; font-size:13px; text-align:right; font-weight:600;">{{ (p*100).toFixed(1) }}%</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- 特征图显示模块 -->
                        <div v-if="testFeatImg" class="test-feat-maps-container" style="background:rgba(255,255,255,0.5); padding:15px; border-radius:12px; border:1px solid var(--panel-border); margin-top:10px;">
                            <h4 style="margin-top:0;">深度网络特征提取过程 (逐层通道激活图谱)</h4>
                            <div style="text-align:center;">
                                <img :src="testFeatImg" style="width:100%; image-rendering:pixelated; border-radius:8px; border:1px solid #eee; box-shadow:0 2px 10px rgba(0,0,0,0.05);">
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    `,
    setup() {
        // --- State ---
        const activeTab = ref('train');
        const hasTorch = ref(true);
        const config = ref({
            lr: 0.01,
            epochs: 3,
            batch_size: 64,
            optimizer: 'Adam',
            loss_fn: 'CrossEntropy',
            layers: [16, 32],
            show_visuals: true
        });
        const isTraining = ref(false);
        const isPaused = ref(false);
        const statusMsg = ref('就绪。点击“开始训练”启动 MNIST 推理机。');
        const lossHistory = ref([]);
        const trainAcc = ref(0);
        const testAcc = ref(0);
        const featImg = ref('');
        const trueLabel = ref(null);
        const probs = ref([]);

        // Testing state
        const isTesting = ref(false);
        const testImg = ref('');
        const testTrueLabel = ref(null);
        const testPrediction = ref(null);
        const testProbs = ref([]);
        const testFeatImg = ref('');

        // Canvas state
        const drawCanvas = ref(null);
        const isDrawing = ref(false);
        let ctx = null;

        const ws = ref(null);

        // --- Methods ---
        const switchTab = (tab) => {
            console.log("Switching CNN tab to:", tab);
            activeTab.value = tab;
        };

        const addLayer = () => { if (config.value.layers.length < 5) config.value.layers.push(64); };
        const removeLayer = (idx) => { if (config.value.layers.length > 1) config.value.layers.splice(idx, 1); };

        const startTraining = async () => {
            try {
                // Reset states
                isTraining.value = true;
                isPaused.value = false;
                lossHistory.value = [];
                trainAcc.value = 0;
                testAcc.value = 0;
                featImg.value = '';
                probs.value = [];

                await connectWS();
                ws.value.send(JSON.stringify({
                    action: 'start',
                    config: config.value
                }));
            } catch (e) {
                isTraining.value = false;
                alert("WebSocket 连接失败: " + e.message);
            }
        };

        const stopTraining = () => {
            if (ws.value) {
                ws.value.send(JSON.stringify({ action: 'stop' }));
                // Close and reopen to ensure back-end task is definitely cleaned up on next start
                ws.value.close();
            }
            // Reset ALL UI states immediately
            isTraining.value = false;
            isPaused.value = false;
            statusMsg.value = '训练已停止并重置';
        };

        const pauseTraining = () => {
            if (ws.value) {
                ws.value.send(JSON.stringify({ action: 'pause' }));
                isPaused.value = true;
                statusMsg.value = '训练已暂停';
            }
        };

        const resumeTraining = () => {
            if (ws.value) {
                ws.value.send(JSON.stringify({ action: 'resume' }));
                isPaused.value = false;
            }
        };

        const connectWS = () => {
            if (ws.value && ws.value.readyState === WebSocket.OPEN) return Promise.resolve();
            return new Promise((resolve, reject) => {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const host = window.location.hostname;
                const port = (window.location.port === '8080' || window.location.port === '') ? '32100' : window.location.port;
                const wsUrl = `${protocol}//${host}:${port}/api/cnn/ws/train`;

                console.log("Connecting to CNN WebSocket:", wsUrl);
                ws.value = new WebSocket(wsUrl);

                ws.value.onopen = () => { resolve(); };
                ws.value.onerror = (e) => {
                    console.error("WS Error Details:", e);
                    reject(new Error("连接被拒绝。请确保后端服务在 32100 端口运行，或尝试使用 127.0.0.1 访问。"));
                };

                ws.value.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    handleWSMessage(data);
                };
            });
        };

        const handleWSMessage = (data) => {
            if (data.type === 'status') {
                statusMsg.value = data.msg;
            } else if (data.type === 'progress') {
                lossHistory.value.push(data.loss);
                // Removed limit to show full history
                statusMsg.value = data.msg;
            } else if (data.type === 'visuals') {
                featImg.value = data.feat_img;
                trueLabel.value = data.true_label;
                probs.value = data.probs;
            } else if (data.type === 'finished') {
                trainAcc.value = data.train_acc;
                testAcc.value = data.test_acc;
                isTraining.value = false;
                isPaused.value = false;
                statusMsg.value = '训练完成！准备进行测试...';

                // Auto transition to test tab after a short delay
                setTimeout(() => {
                    activeTab.value = 'test';
                    testRandom(); // Trigger one random test automatically
                }, 1500);

            } else if (data.type === 'stopped') {
                isTraining.value = false;
                isPaused.value = false;
                statusMsg.value = '训练已终止';
            } else if (data.type === 'error') {
                alert("错误: " + data.msg);
                isTraining.value = false;
                isPaused.value = false;
            }
        };

        // Testing
        const testRandom = async () => {
            try {
                isTesting.value = true;
                const res = await axios.post('/api/cnn/test/random');
                updateTestResults(res.data);
            } catch (e) {
                alert("测试失败");
            } finally {
                isTesting.value = false;
            }
        };

        const uploadTestImg = async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            const formData = new FormData();
            formData.append('file', file);
            try {
                const res = await axios.post('/api/cnn/test/upload', formData);
                updateTestResults(res.data);
            } catch (e) {
                alert("上传识别失败");
            }
        };

        const updateTestResults = (data) => {
            testImg.value = data.image_b64;
            testTrueLabel.value = data.true_label !== undefined ? data.true_label : null;
            testPrediction.value = data.prediction;
            testProbs.value = data.probs;
            testFeatImg.value = data.feat_img || '';
        };

        // --- Canvas Drawing Logic ---
        const initCanvas = () => {
            if (!drawCanvas.value) return;
            ctx = drawCanvas.value.getContext('2d');
            ctx.fillStyle = "black";
            ctx.fillRect(0, 0, 200, 200);
            ctx.strokeStyle = "white";
            ctx.lineWidth = 15;
            ctx.lineCap = "round";
            ctx.lineJoin = "round";
        };

        const clearCanvas = () => {
            if (ctx) {
                ctx.fillStyle = "black";
                ctx.fillRect(0, 0, 200, 200);
            }
        };

        const getPos = (e) => {
            const rect = drawCanvas.value.getBoundingClientRect();
            let x, y;
            if (e.touches) {
                x = e.touches[0].clientX - rect.left;
                y = e.touches[0].clientY - rect.top;
            } else {
                x = e.clientX - rect.left;
                y = e.clientY - rect.top;
            }
            return { x, y };
        };

        const startDrawing = (e) => {
            isDrawing.value = true;
            const pos = getPos(e);
            ctx.beginPath();
            ctx.moveTo(pos.x, pos.y);
        };

        const draw = (e) => {
            if (!isDrawing.value) return;
            const pos = getPos(e);
            ctx.lineTo(pos.x, pos.y);
            ctx.stroke();
        };

        const stopDrawing = () => {
            isDrawing.value = false;
        };

        const testCanvas = async () => {
            // Get data from canvas
            const blob = await new Promise(resolve => drawCanvas.value.toBlob(resolve, 'image/png'));
            const formData = new FormData();
            formData.append('file', blob, 'drawing.png');
            try {
                const res = await axios.post('/api/cnn/test/upload', formData);
                updateTestResults(res.data);
            } catch (e) {
                alert("识别失败");
            }
        };

        onMounted(async () => {
            try {
                const res = await axios.get('/api/cnn/status');
                hasTorch.value = res.data.has_torch;
            } catch (e) { }

            nextTick(() => {
                initCanvas();
            });
        });

        watch(activeTab, (newTab) => {
            if (newTab === 'test') {
                nextTick(() => initCanvas());
            }
        });

        onUnmounted(() => {
            if (ws.value) ws.value.close();
        });

        // Computed
        const lossPath = computed(() => {
            if (!lossHistory.value.length) return "";
            // SVG coordinate system 625x250 (1:2.5 ratio)
            // X range: 45 to 605
            // Y range: 220 (Loss 0) to 20 (Loss 2.0)
            const xMin = 45, xMax = 605;
            const yZero = 220, yMaxLoss = 2.0;
            const yRange = 200; // 220 - 20

            return lossHistory.value.map((val, i) => {
                const x = xMin + (i / Math.max(lossHistory.value.length - 1, 1)) * (xMax - xMin);
                const ratio = Math.min(val / yMaxLoss, 1.0);
                const y = yZero - (ratio * yRange);
                return `${x.toFixed(1)},${y.toFixed(1)}`;
            }).join(" ");
        });

        return {
            activeTab, hasTorch, config, isTraining, isPaused, statusMsg,
            trainAcc, testAcc, featImg, trueLabel, probs, lossHistory,
            lossPath,
            startTraining, stopTraining, pauseTraining, resumeTraining,
            addLayer, removeLayer, switchTab,
            testRandom, uploadTestImg, testImg, testTrueLabel, testPrediction, testProbs, testFeatImg,
            drawCanvas, startDrawing, draw, stopDrawing, clearCanvas, testCanvas
        };
    }
}
