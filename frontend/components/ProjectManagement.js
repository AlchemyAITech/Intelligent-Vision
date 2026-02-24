import { ref, computed, nextTick } from 'vue';

export default {
    name: 'ProjectManagement',
    template: `
    <div style="padding: 24px; height: 100%; display: flex; flex-direction: column; box-sizing: border-box;">
        <div v-if="!activeProject">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px;">
                <h2 style="font-size: 24px; font-weight: bold; color: #82318E;">模型项目工程仓</h2>
                <button style="padding: 10px 20px; background: #82318E; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: bold; box-shadow: 0 4px 6px rgba(130,49,142,0.2);" @click="showCreateModal = true">+ 新建项目工程</button>
            </div>

            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px;">
                <!-- Project Cards -->
                <div v-for="proj in projects" :key="proj.id" style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); border: 1px solid #f0e6f5; cursor: pointer; transition: transform 0.2s;" @click="openProject(proj)" onmouseover="this.style.transform='translateY(-5px)'" onmouseout="this.style.transform='none'">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                        <h3 style="font-weight: bold; font-size: 18px; color: #2d3748; margin: 0;">{{ proj.name }}</h3>
                        <span style="font-size: 12px; padding: 4px 8px; border-radius: 4px; background: #faf5ff; color: #6b46c1; font-weight: bold;">{{ proj.type }}</span>
                    </div>
                    <p style="font-size: 14px; color: #718096; margin-bottom: 16px; min-height: 40px;">{{ proj.desc }}</p>
                    <div style="font-size: 12px; color: #a0aec0;">最后更新: {{ proj.updatedAt }}</div>
                </div>
            </div>

            <!-- Create Modal -->
            <div v-if="showCreateModal" style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(0,0,0,0.4); display: flex; align-items: center; justify-content: center; z-index: 50;">
                <div style="background: white; border-radius: 16px; padding: 32px; width: 400px; box-shadow: 0 20px 40px rgba(0,0,0,0.2);">
                    <h3 style="font-weight: bold; font-size: 20px; margin-bottom: 24px; color: #2d3748;">新建工程</h3>
                    <div style="margin-bottom: 16px;">
                        <label style="display: block; font-size: 14px; color: #4a5568; margin-bottom: 8px; font-weight: bold;">项目名称</label>
                        <input v-model="newProjectForm.name" type="text" style="width: 100%; border: 1px solid #e2e8f0; padding: 10px; border-radius: 8px; outline: none; box-sizing: border-box;" placeholder="如：肝结节筛查大模型">
                    </div>
                    <div style="margin-bottom: 24px;">
                        <label style="display: block; font-size: 14px; color: #4a5568; margin-bottom: 8px; font-weight: bold;">应用范式</label>
                        <select v-model="newProjectForm.type" style="width: 100%; border: 1px solid #e2e8f0; padding: 10px; border-radius: 8px; outline: none; box-sizing: border-box; background: white;">
                            <option value="Classification">图像分类 (Classification)</option>
                            <option value="Detection">目标检测 (Detection)</option>
                            <option value="Segmentation">实例分割 (Segmentation)</option>
                            <option value="Keypoint">关键点检测 (Keypoint)</option>
                        </select>
                    </div>
                    <div style="display: flex; justify-content: flex-end; gap: 12px; margin-top: 32px;">
                        <button style="padding: 10px 16px; background: #edf2f7; border-radius: 8px; color: #4a5568; font-weight: bold; border: none; cursor: pointer;" @click="showCreateModal = false">取消</button>
                        <button style="padding: 10px 16px; background: #82318E; border-radius: 8px; color: white; font-weight: bold; border: none; cursor: pointer;" @click="createProject">创建项目</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Project Detail View (Canvas/Flow) -->
        <div v-else-if="!isSandboxOpen" style="flex: 1; display: flex; flex-direction: column;">
            <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid #e2e8f0;">
                <button style="color: #82318E; font-weight: bold; background: none; border: none; cursor: pointer; display: flex; align-items: center; font-size: 16px;" @click="activeProject = null">
                    ◀ 返回列表
                </button>
                <h2 style="font-size: 20px; font-weight: bold; margin: 0;">{{ activeProject.name }}</h2>
                <span style="font-size: 12px; padding: 4px 8px; border-radius: 4px; background: #faf5ff; color: #6b46c1; font-weight: bold;">{{ activeProject.type }}</span>
            </div>

            <!-- Pipeline UI -->
            <div style="flex: 1; background: #fafafc; border-radius: 16px; padding: 32px; border: 1px dashed #cbd5e0; display: flex; flex-direction: column; align-items: center;">
                <h3 style="color: #a0aec0; font-size: 14px; margin-bottom: 40px; text-transform: uppercase; letter-spacing: 2px; font-weight: bold;">Ultralytics 训练链路设计域</h3>
                
                <div style="display: flex; align-items: center; justify-content: center; gap: 32px; width: 100%; max-width: 900px;">
                    <!-- Dataset Node -->
                    <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); border-left: 6px solid #4299e1; min-width: 250px; flex: 1;">
                        <div style="font-size: 12px; color: #4299e1; font-weight: bold; margin-bottom: 8px;">输入节点 (Input)</div>
                        <div style="font-weight: bold; margin-bottom: 16px; font-size: 18px;">数据集绑定</div>
                        <select style="width: 100%; border: 1px solid #e2e8f0; padding: 10px; border-radius: 8px; font-size: 14px; background: white;" v-model="selectedDataset">
                            <option value="">-- 选择已挂载数据 --</option>
                            <option value="ds1">肝部 CT 资产包 (按类分目录)</option>
                            <option value="ds2">示例 COCO 格式数据包</option>
                        </select>
                    </div>

                    <div style="color: #cbd5e0; font-size: 32px;">➔</div>

                    <!-- Model Node -->
                    <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); border-left: 6px solid #9f7aea; min-width: 250px; flex: 1;">
                        <div style="font-size: 12px; color: #9f7aea; font-weight: bold; margin-bottom: 8px;">计算节点 (Compute)</div>
                        <div style="font-weight: bold; margin-bottom: 16px; font-size: 18px;">模型架构挂载</div>
                        <select style="width: 100%; border: 1px solid #e2e8f0; padding: 10px; border-radius: 8px; font-size: 14px; background: white;" v-model="selectedModel">
                            <option value="">-- 选择底层算法库 --</option>
                            <option value="yolov8n">YOLOv8n (微型)</option>
                            <option value="yolov8s">YOLOv8s (小型)</option>
                            <option value="resnet50">ResNet50 (仅分类)</option>
                        </select>
                    </div>

                    <div style="color: #cbd5e0; font-size: 32px;">➔</div>

                    <!-- Output Node -->
                    <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); border-left: 6px solid #48bb78; min-width: 250px; flex: 1; display:flex; flex-direction: column; justify-content: center;">
                        <div style="font-size: 12px; color: #48bb78; font-weight: bold; margin-bottom: 8px;">输出节点 (Output)</div>
                        <div style="font-weight: bold; margin-bottom: 16px; font-size: 18px;">训练沙箱流向</div>
                        <button style="width: 100%; background: #f0fff4; color: #38a169; border: 1px solid #9ae6b4; padding: 12px; border-radius: 8px; font-weight: bold; cursor: pointer; transition: all 0.2s;" @click="enterSandbox" onmouseover="this.style.background='#c6f6d5'" onmouseout="this.style.background='#f0fff4'">
                            进入大屏沙箱 🚀
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Training Sandbox (Phase 4 & 5) -->
        <div v-else style="flex: 1; display: flex; flex-direction: column; background: #0f172a; border-radius: 16px; color: white; padding: 24px; overflow: hidden; box-shadow: 0 20px 40px rgba(0,0,0,0.3);">
            <!-- Header -->
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; border-bottom: 1px solid #334155; padding-bottom: 16px;">
                <div style="display: flex; align-items: center; gap: 16px;">
                    <button style="color: #94a3b8; font-weight: bold; background: none; border: none; cursor: pointer; font-size: 16px;" @click="exitSandbox">
                        ◀ 返回管线
                    </button>
                    <h2 style="font-size: 20px; font-weight: bold; margin: 0; color: #f8fafc;">🔥 异构加速训练控制台</h2>
                    <span style="font-size: 12px; padding: 4px 8px; border-radius: 4px; background: #1e293b; border: 1px solid #334155; color: #94a3b8;">{{ activeProject?.name }} ⇋ {{ selectedModel }}</span>
                </div>
                <div style="display: flex; gap: 12px; align-items: center;">
                    <button :class="['sandbox-nav', sandboxTab === 'training' ? 'active' : '']" @click="changeSandboxTab('training')">🔥 算力大盘</button>
                    <button :class="['sandbox-nav', sandboxTab === 'cam' ? 'active' : '']" @click="changeSandboxTab('cam')">🧪 可解释性探测</button>
                    <button :class="['sandbox-nav', sandboxTab === 'pca' ? 'active' : '']" @click="changeSandboxTab('pca')">🌐 PCA 高维聚类雷达</button>
                </div>
            </div>

            <!-- Tab 1: Training -->
            <div v-show="sandboxTab === 'training'" style="display: flex; gap: 24px; flex: 1; overflow: hidden;">
                <!-- 左侧: 控制板 -->
                <div style="flex: 1; max-width: 300px; background: #1e293b; border-radius: 12px; padding: 20px; display: flex; flex-direction: column;">
                    <h3 style="font-size: 16px; color: #e2e8f0; margin-bottom: 20px; font-weight: bold;">算力网络下发参数</h3>
                    
                    <div style="margin-bottom: 16px;">
                        <label style="display: block; font-size: 13px; color: #94a3b8; margin-bottom: 8px;">迭代轮次 (Epochs)</label>
                        <input type="number" v-model.number="trainConfig.epochs" style="width: 100%; background: #0f172a; border: 1px solid #334155; color: white; padding: 8px; border-radius: 6px; box-sizing: border-box;" :disabled="trainingStatus === 'running'">
                    </div>
                    
                    <div style="margin-bottom: 16px;">
                        <label style="display: block; font-size: 13px; color: #94a3b8; margin-bottom: 8px;">批次大小 (Batch Size)</label>
                        <input type="number" v-model.number="trainConfig.batch" style="width: 100%; background: #0f172a; border: 1px solid #334155; color: white; padding: 8px; border-radius: 6px; box-sizing: border-box;" :disabled="trainingStatus === 'running'">
                    </div>

                    <div style="margin-bottom: 24px;">
                        <label style="display: block; font-size: 13px; color: #94a3b8; margin-bottom: 8px;">优化器 (Optimizer)</label>
                        <select v-model="trainConfig.optimizer" style="width: 100%; background: #0f172a; border: 1px solid #334155; color: white; padding: 8px; border-radius: 6px; box-sizing: border-box; appearance: none;" :disabled="trainingStatus === 'running'">
                            <option value="auto">Auto (智能决断)</option>
                            <option value="SGD">SGD (梯度下降)</option>
                            <option value="AdamW">AdamW (自适应增强)</option>
                        </select>
                    </div>

                    <div style="flex: 1;"></div> <!-- Spacer -->

                    <button v-if="trainingStatus !== 'running'" @click="startTraining" style="width: 100%; padding: 12px; background: linear-gradient(135deg, #a21caf, #6b21a8); color: white; border: none; border-radius: 8px; font-weight: bold; cursor: pointer; box-shadow: 0 4px 15px rgba(162, 28, 175, 0.4); margin-bottom: 10px;">
                        🚀 下发训练指令
                    </button>
                    <button v-else style="width: 100%; padding: 12px; background: transparent; border: 1px dashed #4ade80; color: #4ade80; border-radius: 8px; font-weight: bold; cursor: not-allowed; margin-bottom: 10px;">
                        <span class="pulse">●</span> 正在计算中...
                    </button>
                    <!-- ONNX Output -->
                    <button style="width: 100%; padding: 12px; background: #334155; color: #cbd5e0; border: none; border-radius: 8px; font-weight: bold; cursor: pointer; transition: background 0.2s;" @click="exportOnnx">
                        📦 一键导出 ONNX 跨端张量
                    </button>
                </div>

                <!-- 右侧: 图表区 -->
                <div style="flex: 3; display: flex; flex-direction: column; gap: 20px;">
                    <div style="flex: 2; display: flex; gap: 20px;">
                        <div style="flex: 1; background: #1e293b; border-radius: 12px; padding: 16px; border: 1px solid #334155; position: relative;">
                            <h4 style="font-size: 14px; color: #cbd5e0; margin: 0 0 10px 0;">📉 实时 Box Loss 衰减</h4>
                            <div id="chart-loss" style="width: 100%; height: 90%;"></div>
                        </div>
                        <div style="flex: 1; background: #1e293b; border-radius: 12px; padding: 16px; border: 1px solid #334155; position: relative;">
                            <h4 style="font-size: 14px; color: #cbd5e0; margin: 0 0 10px 0;">🎯 验证集 mAP@50 精度</h4>
                            <div id="chart-map" style="width: 100%; height: 90%;"></div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Tab 2: Grad CAM -->
            <div v-show="sandboxTab === 'cam'" style="display: flex; flex-direction: column; flex: 1; align-items: center; justify-content: center; gap: 24px; background: #1e293b; border-radius: 12px; border: 1px solid #334155;">
                <h3 style="color: #e2e8f0; font-size: 18px; font-weight: bold; margin: 0;">🔮 内网透视 / Grad-CAM 注意力焦点剥离测试</h3>
                <p style="color: #94a3b8; font-size: 14px; max-width: 600px; text-align: center;">请挂载一张新的测试图片。我们将贯穿当前模型的深层卷积神经网络，并通过伪彩色热力图显示大模型判定此目标时聚焦的最佳判别特征区。</p>
                <div style="display: flex; gap: 20px; align-items: stretch; width: 60%; max-width: 800px; min-height: 350px;">
                    <div style="flex: 1; border: 2px dashed #475569; border-radius: 12px; display: flex; align-items: center; justify-content: center; position: relative; cursor: pointer;" @click="triggerCamUpload">
                        <span v-if="!camInputUrl" style="color: #94a3b8; font-weight: bold;">[点击挂载] 本地检验图像</span>
                        <img v-else :src="camInputUrl" style="max-width: 100%; max-height: 100%; border-radius: 8px; object-fit: contain;">
                        <input type="file" ref="camFileRef" style="display: none;" @change="handleCamUpload" accept="image/*">
                    </div>
                    <div style="color: #475569; font-size: 40px; display: flex; align-items: center;">➔</div>
                    <div style="flex: 1; border: 1px solid #334155; background: #0f172a; border-radius: 12px; display: flex; align-items: center; justify-content: center;">
                        <span v-if="!camResultUrl && !isCamLoading" style="color: #475569;">暂无解析特征层...</span>
                        <div v-if="isCamLoading" class="pulse" style="color: #a21caf; font-weight: bold;">正在穿透推理神经网路...</div>
                        <img v-if="camResultUrl && !isCamLoading" :src="camResultUrl" style="max-width: 100%; max-height: 100%; border-radius: 8px; object-fit: contain; box-shadow: 0 0 20px rgba(220, 38, 38, 0.4);">
                    </div>
                </div>
                <button v-if="camInputUrl && !isCamLoading" style="padding: 12px 32px; background: #4ade80; color: #064e3b; border: none; border-radius: 8px; font-weight: bold; cursor: pointer;" @click="executeCam">开启热力扫描靶向探测网络</button>
            </div>

            <!-- Tab 3: PCA Cluster -->
            <div v-show="sandboxTab === 'pca'" style="display: flex; flex-direction: column; flex: 1; gap: 24px;">
                <div style="background: #1e293b; padding: 24px; border-radius: 12px; border: 1px solid #334155; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h3 style="color: #e2e8f0; font-size: 18px; font-weight: bold; margin: 0 0 8px 0;">🌐 高维特征聚类空间解析仪 (PCA)</h3>
                        <p style="color: #94a3b8; font-size: 14px; margin: 0;">一键对项目资产包内的数百张影像进行前向传播提取全联接层向量，使用高阶降维算法 (PCA) 在二维散点图验证类的分离度。</p>
                    </div>
                    <button style="padding: 12px 24px; background: #3b82f6; color: white; border: none; border-radius: 8px; font-weight: bold; cursor: pointer;" @click="executePCA" :disabled="isPcaLoading">
                        <span v-if="!isPcaLoading">⚡ 全量沙盘投射演算</span>
                        <span v-else class="pulse">🔬 正在汇聚散斑降维...</span>
                    </button>
                </div>
                <div style="flex: 1; background: #1e293b; border-radius: 12px; border: 1px solid #334155; position: relative;">
                    <div id="chart-pca" style="width: 100%; height: 100%;"></div>
                </div>
            </div>
        </div>

        <style>
            .pulse {
                animation: pulse-animation 1.5s infinite;
            }
            @keyframes pulse-animation {
                0% { opacity: 1; }
                50% { opacity: 0.4; }
                100% { opacity: 1; }
            }
            .sandbox-nav {
                background: transparent;
                border: 1px solid #334155;
                color: #94a3b8;
                padding: 6px 14px;
                border-radius: 20px;
                font-size: 13px;
                cursor: pointer;
                transition: all 0.2s;
            }
            .sandbox-nav.active {
                background: rgba(162, 28, 175, 0.2);
                border-color: #a21caf;
                color: #e879f9;
                font-weight: bold;
            }
            .sandbox-nav:hover:not(.active) {
                background: #1e293b;
                color: #cbd5e0;
            }
        </style>
    </div>
    `,
    setup() {
        const projects = ref([
            { id: 1, name: '演示项目: 细胞形态分类', type: 'Classification', desc: '红细胞与白细胞微观显微镜图像分类器。', updatedAt: '2026-02-23' },
            { id: 2, name: '骨折区域检测检测', type: 'Detection', desc: 'X光片骨折断裂区域边界框识别。', updatedAt: '2026-02-24' }
        ]);

        const activeProject = ref(null);
        const showCreateModal = ref(false);
        const newProjectForm = ref({ name: '', type: 'Classification' });

        const selectedDataset = ref('');
        const selectedModel = ref('');

        const openProject = (proj) => {
            activeProject.value = proj;
            isSandboxOpen.value = false;
        };

        const createProject = () => {
            if (!newProjectForm.value.name.trim()) return;
            projects.value.unshift({
                id: Date.now(),
                name: newProjectForm.value.name,
                type: newProjectForm.value.type,
                desc: '新建未配置项目。',
                updatedAt: new Date().toISOString().split('T')[0]
            });
            showCreateModal.value = false;
            newProjectForm.value = { name: '', type: 'Classification' };
        };

        const isSandboxOpen = ref(false);
        const sandboxTab = ref('training');
        const trainingStatus = ref('idle');
        const trainConfig = ref({ epochs: 10, batch: 8, optimizer: 'auto' });

        let lossChart = null;
        let mapChart = null;
        let pcaChart = null;
        let lossData = [];
        let mapData = [];
        let curWebSocket = null;

        const changeSandboxTab = (tab) => {
            sandboxTab.value = tab;
            nextTick(() => {
                if (tab === 'training') initCharts();
                if (tab === 'pca' && pcaChart) pcaChart.resize();
            });
        };

        const initCharts = () => {
            if (!window.echarts) {
                console.error("Echarts hasn't loaded.");
                return;
            }
            if (!lossChart) {
                lossChart = window.echarts.init(document.getElementById('chart-loss'));
                mapChart = window.echarts.init(document.getElementById('chart-map'));
            }

            const commonOptions = {
                grid: { left: 40, right: 20, top: 20, bottom: 30 },
                xAxis: { type: 'category', data: [], axisLine: { lineStyle: { color: '#475569' } }, axisLabel: { color: '#94a3b8' } },
                yAxis: { type: 'value', splitLine: { lineStyle: { color: '#334155' } }, axisLabel: { color: '#94a3b8' } },
                tooltip: { trigger: 'axis', backgroundColor: '#1e293b', borderColor: '#475569', textStyle: { color: '#f8fafc' } }
            };

            lossChart.setOption({
                ...commonOptions,
                series: [{ name: 'Box Loss', type: 'line', data: [], smooth: true, lineStyle: { color: '#ef4444', width: 3 }, showSymbol: false, itemStyle: { color: '#ef4444' } }]
            });

            mapChart.setOption({
                ...commonOptions,
                series: [{ name: 'mAP@50', type: 'line', data: [], smooth: true, lineStyle: { color: '#3b82f6', width: 3 }, showSymbol: false, itemStyle: { color: '#3b82f6' }, areaStyle: { color: new window.echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: 'rgba(59,130,246,0.3)' }, { offset: 1, color: 'rgba(59,130,246,0)' }]) } }]
            });
        };

        const initPcaChart = (points) => {
            if (!pcaChart) {
                pcaChart = window.echarts.init(document.getElementById('chart-pca'));
            }
            // 按照 label 对 scatter 进行归组
            const seriesData = {};
            points.forEach(p => {
                if (!seriesData[p.label]) seriesData[p.label] = [];
                seriesData[p.label].push([p.x, p.y]);
            });

            const series = Object.keys(seriesData).map(label => {
                return {
                    name: label,
                    type: 'scatter',
                    symbolSize: 8,
                    data: seriesData[label]
                };
            });

            pcaChart.setOption({
                backgroundColor: 'transparent',
                tooltip: { trigger: 'item', formatter: '{a} <br/>({c})' },
                legend: { top: 20, textStyle: { color: '#e2e8f0' } },
                xAxis: { type: 'value', splitLine: { show: false }, axisLine: { lineStyle: { color: '#475569' } } },
                yAxis: { type: 'value', splitLine: { lineStyle: { color: '#334155', type: 'dashed' } }, axisLine: { lineStyle: { color: '#475569' } } },
                series: series,
                color: ['#4ade80', '#fbbf24', '#f87171', '#c084fc']
            });
        };

        const enterSandbox = () => {
            if (!selectedDataset.value || !selectedModel.value) {
                alert('请先完整挂载数据流与算法底座。');
                return;
            }
            isSandboxOpen.value = true;
            nextTick(() => {
                initCharts();
            });
        };

        const exitSandbox = () => {
            isSandboxOpen.value = false;
            // Clean logic
            if (curWebSocket) {
                curWebSocket.close();
            }
        };

        const startTraining = async () => {
            trainingStatus.value = 'running';
            lossData = []; mapData = [];

            // clear chart
            const emptyOption = { xAxis: { data: [] }, series: [{ data: [] }] };
            lossChart.setOption(emptyOption);
            mapChart.setOption(emptyOption);

            const jobId = "job_" + new Date().getTime();

            // 1. Establish WS connection first to catch early logs
            const wsUrl = `ws://localhost:8000/api/training/ws/${jobId}`;
            curWebSocket = new WebSocket(wsUrl);

            curWebSocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.epoch !== undefined) {
                        lossData.push(data.box_loss);
                        mapData.push(data.map50);
                        const xAxisData = Array.from({ length: lossData.length }, (_, i) => `Ep${i + 1}`);

                        lossChart.setOption({ xAxis: { data: xAxisData }, series: [{ data: lossData }] });
                        mapChart.setOption({ xAxis: { data: xAxisData }, series: [{ data: mapData }] });
                    }
                } catch (e) { }
            };

            // 2. Trigger POST Request
            try {
                await axios.post('/api/training/start', {
                    project_name: activeProject.value.name,
                    job_id: jobId,
                    yaml_path: 'dataset.yaml', // mock
                    model_type: selectedModel.value,
                    epochs: trainConfig.value.epochs,
                    batch_size: trainConfig.value.batch,
                    optimizer: trainConfig.value.optimizer
                });
                console.log("[Launch] Command sent tracking job", jobId);
            } catch (err) {
                console.error(err);
                alert("启动失败请检查后端服务");
                trainingStatus.value = 'idle';
            }
        };

        const exportOnnx = async () => {
            try {
                const res = await axios.post(`/api/training/export_onnx/${activeProject.value.name}/job_mock`);
                alert('🚀 ONNX 导出成功，链路：' + res.data.onnx_path);
            } catch (e) {
                alert('环境内暂未发现当前权重的 .pt 固态文件，仅可用于沙盘功能推演');
            }
        };

        // Grad CAM logic
        const camFileRef = ref(null);
        const camFileRaw = ref(null);
        const camInputUrl = ref('');
        const camResultUrl = ref('');
        const isCamLoading = ref(false);

        const triggerCamUpload = () => camFileRef.value.click();

        const handleCamUpload = (e) => {
            if (e.target.files.length > 0) {
                const f = e.target.files[0];
                camFileRaw.value = f;
                camInputUrl.value = URL.createObjectURL(f);
                camResultUrl.value = '';
            }
        };

        const executeCam = async () => {
            if (!camFileRaw.value) return;
            isCamLoading.value = true;
            try {
                const formData = new FormData();
                formData.append('file', camFileRaw.value);
                const res = await axios.post(`/api/analytica/grad_cam?project_name=${activeProject.value.name}`, formData, {
                    headers: { 'Content-Type': 'multipart/form-data' }
                });
                camResultUrl.value = res.data.cam_url;
            } catch (err) {
                alert('可解释探测网络未接通');
            } finally {
                isCamLoading.value = false;
            }
        };

        // PCA Logic
        const isPcaLoading = ref(false);
        const executePCA = async () => {
            isPcaLoading.value = true;
            try {
                const res = await axios.post(`/api/analytica/pca_cluster`, { project_name: activeProject.value.name });
                if (res.data.status === 'success') {
                    initPcaChart(res.data.points);
                }
            } catch (e) {
                console.error(e);
            } finally {
                isPcaLoading.value = false;
            }
        };

        return {
            projects,
            activeProject,
            showCreateModal,
            newProjectForm,
            openProject,
            createProject,
            selectedDataset,
            selectedModel,
            isSandboxOpen,
            sandboxTab,
            changeSandboxTab,
            enterSandbox,
            exitSandbox,
            trainingStatus,
            trainConfig,
            startTraining,
            exportOnnx,
            camFileRef, camInputUrl, camResultUrl, isCamLoading, triggerCamUpload, handleCamUpload, executeCam,
            isPcaLoading, executePCA
        };
    }
};
