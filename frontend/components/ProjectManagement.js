import { ref, computed, nextTick, watch, onMounted } from 'vue';

export default {
    name: 'ProjectManagement',
    template: `
    <div style="padding: 24px; height: 100%; display: flex; flex-direction: column; box-sizing: border-box; position: relative;">
        <div v-if="!activeProject">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px;">
                <h2 style="font-size: 24px; font-weight: bold; color: #82318E;">模型项目工程仓</h2>
                <button style="padding: 10px 20px; background: #82318E; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: bold; box-shadow: 0 4px 6px rgba(130,49,142,0.2);" @click="showCreateModal = true">+ 新建项目工程</button>
            </div>

            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px;">
                <!-- Project Cards -->
                <div v-for="proj in projects" :key="proj.id" style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); border: 1px solid #f0e6f5; cursor: pointer; transition: transform 0.2s; position: relative;" @click="openProject(proj)" onmouseover="this.style.transform='translateY(-5px)'" onmouseout="this.style.transform='none'">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                        <h3 style="font-weight: bold; font-size: 18px; color: #2d3748; margin: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 60%;">{{ proj.name }}</h3>
                        <span style="font-size: 12px; padding: 4px 8px; border-radius: 4px; background: #faf5ff; color: #6b46c1; font-weight: bold;">{{ proj.type }}</span>
                    </div>
                    <p style="font-size: 14px; color: #718096; margin-bottom: 16px; min-height: 40px; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;">{{ proj.desc }}</p>
                    <div style="font-size: 12px; color: #a0aec0; display: flex; justify-content: space-between; align-items: center;">
                        <span>最后更新: {{ proj.updatedAt }}</span>
                        <div style="display: flex; gap: 8px;">
                            <button @click.stop="renameProject(proj)" style="padding: 4px 8px; font-size: 12px; background: transparent; border: 1px solid #e2e8f0; border-radius: 4px; color: #4a5568; cursor: pointer;">重命名</button>
                            <button @click.stop="deleteProject(proj)" style="padding: 4px 8px; font-size: 12px; background: #fed7d7; border: 1px solid #feb2b2; border-radius: 4px; color: #c53030; cursor: pointer;">删除</button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Create Modal -->
            <div v-if="showCreateModal" style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(0,0,0,0.4); display: flex; align-items: center; justify-content: center; z-index: 50;">
                <div style="background: white; border-radius: 16px; padding: 32px; width: 400px; box-shadow: 0 20px 40px rgba(0,0,0,0.2);">
                    <h3 style="font-weight: bold; font-size: 20px; margin-bottom: 24px; color: #2d3748;">新建工程</h3>
                    <div style="margin-bottom: 16px;">
                        <label style="display: block; font-size: 14px; color: #4a5568; margin-bottom: 8px; font-weight: bold;">项目名称</label>
                        <input type="text" v-model="newProjectForm.name" style="width: 100%; border: 1px solid #e2e8f0; padding: 10px; border-radius: 8px; font-size: 14px; box-sizing: border-box;" placeholder="例如：医学影像第一版实验">
                    </div>
                    <div style="margin-bottom: 24px;">
                        <label style="display: block; font-weight: bold; margin-bottom: 8px; font-size: 14px; color: #4a5568;">项目描述</label>
                        <textarea v-model="newProjectForm.desc" style="width: 100%; border: 1px solid #e2e8f0; padding: 10px; border-radius: 8px; font-size: 14px; box-sizing: border-box; resize: vertical;" placeholder="描述此工程目标，如超参数调优记录等"></textarea>
                    </div>
                    <div style="display: flex; justify-content: flex-end; gap: 12px; margin-top: 32px;">
                        <button style="padding: 10px 16px; background: #edf2f7; border-radius: 8px; color: #4a5568; font-weight: bold; border: none; cursor: pointer;" @click="showCreateModal = false">取消</button>
                        <button style="padding: 10px 16px; background: #82318E; border-radius: 8px; color: white; font-weight: bold; border: none; cursor: pointer;" @click="createProject">创建项目</button>
                    </div>
                </div>
            </div>

            <!-- Rename Modal -->
            <div v-if="showRenameModal" style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; z-index: 9999;">
                <div style="background: white; border-radius: 16px; padding: 32px; width: 400px; box-shadow: 0 20px 40px rgba(0,0,0,0.3);">
                    <h3 style="font-weight: bold; font-size: 20px; margin-bottom: 24px; color: #2d3748;">重命名工程仓</h3>
                    <div style="margin-bottom: 24px;">
                        <label style="display: block; font-size: 14px; color: #4a5568; margin-bottom: 8px; font-weight: bold;">新项目名称</label>
                        <input type="text" v-model="renameForm.name" style="width: 100%; border: 1px solid #e2e8f0; padding: 10px; border-radius: 8px; font-size: 14px; box-sizing: border-box;">
                    </div>
                    <div style="display: flex; justify-content: flex-end; gap: 12px; margin-top: 32px;">
                        <button style="padding: 10px 16px; background: #edf2f7; border-radius: 8px; color: #4a5568; font-weight: bold; border: none; cursor: pointer;" @click="showRenameModal = false">取消</button>
                        <button style="padding: 10px 16px; background: #82318E; border-radius: 8px; color: white; font-weight: bold; border: none; cursor: pointer;" @click="confirmRenameProject">确认重命名</button>
                    </div>
                </div>
            </div>

            <!-- Delete Confirm Modal -->
            <div v-if="showDeleteModal" style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; z-index: 9999;">
                <div style="background: white; border-radius: 16px; padding: 32px; width: 400px; box-shadow: 0 20px 40px rgba(0,0,0,0.3);">
                    <h3 style="font-weight: bold; font-size: 20px; margin-bottom: 16px; color: #e53e3e;">警告：极端危险操作</h3>
                    <p style="color: #4a5568; font-size: 14px; margin-bottom: 24px; line-height: 1.5;">
                        确定要彻底摧毁工程仓 <strong style="color: #e53e3e;">"{{ deleteTarget?.name }}"</strong> 吗？<br>
                        此操作不可逆转，将清空内部生成的所有 AI 权重！
                    </p>
                    <div style="display: flex; justify-content: flex-end; gap: 12px;">
                        <button style="padding: 10px 16px; background: #edf2f7; border-radius: 8px; color: #4a5568; font-weight: bold; border: none; cursor: pointer;" @click="showDeleteModal = false">取消</button>
                        <button style="padding: 10px 16px; background: #e53e3e; border-radius: 8px; color: white; font-weight: bold; border: none; cursor: pointer;" @click="confirmDeleteProject">确定摧毁</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Project Detail View (Canvas/Flow) -->
        <div v-else-if="activeProject && !isSandboxOpen && !isModelManagementOpen" style="flex: 1; display: flex; flex-direction: column;">
            <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid #e2e8f0;">
                <button style="color: #82318E; font-weight: bold; background: none; border: none; cursor: pointer; display: flex; align-items: center; font-size: 16px;" @click="activeProject = null">
                    ◀ 返回列表
                </button>
                <h2 style="font-size: 20px; font-weight: bold; margin: 0;">{{ activeProject.name }}</h2>
            </div>

            <!-- Global Toast Notification -->
            <div v-if="toastMessage" :style="{ position: 'fixed', top: '24px', left: '50%', transform: 'translateX(-50%)', padding: '12px 24px', borderRadius: '8px', background: toastType === 'error' ? '#e53e3e' : '#38a169', color: 'white', fontWeight: 'bold', boxShadow: '0 10px 15px rgba(0,0,0,0.2)', zIndex: 99999, transition: 'all 0.3s ease', display: 'flex', alignItems: 'center', gap: '8px' }">
                <span v-if="toastType === 'success'">✅</span>
                <span v-if="toastType === 'error'">❌</span>
                {{ toastMessage }}
            </div>

            <!-- Pipeline UI -->
            <div style="flex: 1; background: #fafafc; border-radius: 16px; padding: 32px; border: 1px dashed #cbd5e0; display: flex; flex-direction: column; align-items: center;">
                <h3 style="color: #a0aec0; font-size: 14px; margin-bottom: 40px; text-transform: uppercase; letter-spacing: 2px; font-weight: bold;">Ultralytics 训练链路设计域</h3>
                
                <div style="display: flex; align-items: center; justify-content: center; gap: 32px; width: 100%; max-width: 900px;">
                    <!-- Phase 1: Data Binding -->
                    <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); border-left: 6px solid #4299e1; min-width: 200px; flex: 1; display:flex; flex-direction: column;">
                        <div style="font-size: 12px; color: #4299e1; font-weight: bold; margin-bottom: 8px;">阶段一 (Phase 1)</div>
                        <div style="font-weight: bold; margin-bottom: 16px; font-size: 18px;">数据绑定</div>
                        <select style="width: 100%; border: 1px solid #e2e8f0; padding: 10px; border-radius: 8px; font-size: 14px; background: white; margin-bottom: 8px;" v-model="selectedDataset">
                            <option value="">-- 选择已挂载数据 --</option>
                            <option v-for="ds in datasetList" :key="ds.name" :value="ds.name">{{ ds.name }} ({{ ds.type }})</option>
                        </select>
                        <select v-if="selectedDataset" style="width: 100%; border: 1px solid #e2e8f0; padding: 10px; border-radius: 8px; font-size: 14px; background: white;" v-model="selectedDataVersion">
                            <option value="">-- 选择标注版本 --</option>
                            <option v-for="ver in availableDataVersions" :key="ver.version" :value="ver.version">{{ ver.version }} {{ ver.description ? '(' + ver.description + ')' : '' }}</option>
                        </select>
                    </div>

                    <div style="color: #cbd5e0; font-size: 24px;">➔</div>

                    <!-- Phase 2: Model Training -->
                    <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); border-left: 6px solid #9f7aea; min-width: 200px; flex: 1; display:flex; flex-direction: column;">
                        <div style="font-size: 12px; color: #9f7aea; font-weight: bold; margin-bottom: 8px;">阶段二 (Phase 2)</div>
                        <div style="font-weight: bold; margin-bottom: 16px; font-size: 18px;">模型训练</div>
                        <select style="width: 100%; border: 1px solid #e2e8f0; padding: 10px; border-radius: 8px; font-size: 14px; background: white; margin-bottom: 16px;" v-model="selectedModel">
                            <option value="">-- 选择算法底座 --</option>
                            <option value="yolov8n-cls.pt">YOLOv8n (微型分类)</option>
                            <option value="yolov8s-cls.pt">YOLOv8s (小型分类)</option>
                            <option value="resnet50.pt">ResNet50 (仅分类)</option>
                        </select>
                        <button :disabled="!selectedDataset || !selectedDataVersion || !selectedModel" style="width: 100%; background: #faf5ff; color: #805ad5; border: 1px solid #d6bcfa; padding: 10px; border-radius: 8px; font-weight: bold; cursor: pointer; transition: all 0.2s;" @click="enterSandbox('training')" onmouseover="if(!this.disabled) this.style.background='#e9d8fd'" onmouseout="if(!this.disabled) this.style.background='#faf5ff'">
                            进入大屏沙箱 🚀
                        </button>
                    </div>

                    <div style="color: #cbd5e0; font-size: 24px;">➔</div>

                    <!-- Phase 3: Model Test -->
                    <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); border-left: 6px solid #48bb78; min-width: 200px; flex: 1; display:flex; flex-direction: column; justify-content: center;">
                        <div style="font-size: 12px; color: #48bb78; font-weight: bold; margin-bottom: 8px;">阶段三 (Phase 3)</div>
                        <div style="font-weight: bold; margin-bottom: 16px; font-size: 18px;">模型测试</div>
                        <button style="width: 100%; background: #f0fff4; color: #38a169; border: 1px solid #9ae6b4; padding: 12px; border-radius: 8px; font-weight: bold; cursor: pointer; transition: all 0.2s;" @click="enterSandbox('cam')" onmouseover="this.style.background='#c6f6d5'" onmouseout="this.style.background='#f0fff4'">
                            进入评估基站 🧪
                        </button>
                    </div>

                    <div style="color: #cbd5e0; font-size: 24px;">➔</div>

                    <!-- Phase 4: Model Management -->
                    <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); border-left: 6px solid #ed8936; min-width: 200px; flex: 1; display:flex; flex-direction: column; justify-content: center;">
                        <div style="font-size: 12px; color: #ed8936; font-weight: bold; margin-bottom: 8px;">阶段四 (Phase 4)</div>
                        <div style="font-weight: bold; margin-bottom: 16px; font-size: 18px;">模型管理</div>
                        <button style="width: 100%; background: #fffaf0; color: #dd6b20; border: 1px solid #fbd38d; padding: 12px; border-radius: 8px; font-weight: bold; cursor: pointer; transition: all 0.2s;" @click="openModelManagement" onmouseover="this.style.background='#feebc8'" onmouseout="this.style.background='#fffaf0'">
                            模型版本中枢 ⚙️
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Training Sandbox (Phase 2, 3, 4) -->
        <div v-else style="flex: 1; display: flex; flex-direction: column; background: #ffffff; border-radius: 16px; color: #2d3748; padding: 24px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.05); border: 1px solid #e2e8f0;">
            <!-- Header -->
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; border-bottom: 1px solid #e2e8f0; padding-bottom: 16px;">
                <div style="display: flex; align-items: center; gap: 16px;">
                    <button style="color: #4a5568; font-weight: bold; background: none; border: none; cursor: pointer; font-size: 16px; display: flex; align-items: center;" @click="exitSandbox">
                        ◀ 返回管线
                    </button>
                    <h2 style="font-size: 20px; font-weight: bold; margin: 0; color: #1a202c;">🔥 异构加速训练控制台</h2>
                    <select v-model="selectedRunForTesting" style="font-size: 12px; padding: 4px 24px 4px 8px; border-radius: 4px; background: #edf2f7; border: 1px solid #cbd5e0; color: #4a5568; outline: none; cursor: pointer; appearance: none; background-image: url('data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22292.4%22%20height%3D%22292.4%22%3E%3Cpath%20fill%3D%22%234a5568%22%20d%3D%22M287%2069.4a17.6%2017.6%200%200%200-13-5.4H18.4c-5%200-9.3%201.8-12.9%205.4A17.6%2017.6%200%200%200%200%2082.2c0%205%201.8%209.3%205.4%2012.9l128%20127.9c3.6%203.6%207.8%205.4%2012.8%205.4s9.2-1.8%2012.8-5.4L287%2095c3.5-3.5%205.4-7.8%205.4-12.8%200-5-1.9-9.2-5.5-12.8z%22%2F%3E%3C%2Fsvg%3E'); background-repeat: no-repeat; background-position: right 8px top 50%; background-size: 8px auto; min-width: 150px;">
                        <option value="" disabled v-if="modelRuns.length === 0">暂无训练完成的版本</option>
                        <option v-for="run in modelRuns" :key="run.name" :value="run.name">{{ activeProject?.name }} ⇋ {{ run.name }}</option>
                    </select>
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
                <div style="flex: 1; max-width: 300px; background: #f8fafc; border-radius: 12px; padding: 20px; display: flex; flex-direction: column; border: 1px solid #e2e8f0;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                        <h3 style="font-size: 16px; color: #2d3748; margin: 0; font-weight: bold;">算力网络下发参数</h3>
                        <span style="color: #e53e3e; font-size: 13px; font-weight: bold;">超参数</span>
                    </div>
                    
                    <div style="margin-bottom: 16px;">
                        <label style="display: block; font-size: 13px; color: #4a5568; margin-bottom: 8px;">迭代轮次 (Epochs)</label>
                        <input type="number" v-model.number="trainConfig.epochs" style="width: 100%; background: white; border: 1px solid #cbd5e0; color: #2d3748; padding: 8px; border-radius: 6px; box-sizing: border-box;" :disabled="trainingStatus === 'running'">
                    </div>
                    
                    <div style="margin-bottom: 16px;">
                        <label style="display: block; font-size: 13px; color: #4a5568; margin-bottom: 8px;">批次大小 (Batch Size)</label>
                        <input type="number" v-model.number="trainConfig.batch" style="width: 100%; background: white; border: 1px solid #cbd5e0; color: #2d3748; padding: 8px; border-radius: 6px; box-sizing: border-box;" :disabled="trainingStatus === 'running'">
                    </div>

                    <div style="margin-bottom: 16px;">
                        <label style="display: block; font-size: 13px; color: #4a5568; margin-bottom: 8px;">图像大小 (Image Size)</label>
                        <input type="number" v-model.number="trainConfig.imgsz" style="width: 100%; background: white; border: 1px solid #cbd5e0; color: #2d3748; padding: 8px; border-radius: 6px; box-sizing: border-box;" :disabled="trainingStatus === 'running'">
                    </div>

                    <div style="margin-bottom: 16px;">
                        <label style="display: block; font-size: 13px; color: #4a5568; margin-bottom: 8px;">初始学习率 (LR0)</label>
                        <input type="number" step="0.001" v-model.number="trainConfig.lr0" style="width: 100%; background: white; border: 1px solid #cbd5e0; color: #2d3748; padding: 8px; border-radius: 6px; box-sizing: border-box;" :disabled="trainingStatus === 'running'">
                    </div>

                    <div style="margin-bottom: 16px;">
                        <label style="display: block; font-size: 13px; color: #4a5568; margin-bottom: 8px;">早停轮数 (Patience)</label>
                        <input type="number" v-model.number="trainConfig.patience" style="width: 100%; background: white; border: 1px solid #cbd5e0; color: #2d3748; padding: 8px; border-radius: 6px; box-sizing: border-box;" :disabled="trainingStatus === 'running'">
                    </div>

                    <div style="margin-bottom: 24px;">
                        <label style="display: block; font-size: 13px; color: #4a5568; margin-bottom: 8px;">优化器 (Optimizer)</label>
                        <select v-model="trainConfig.optimizer" style="width: 100%; background: white; border: 1px solid #cbd5e0; color: #2d3748; padding: 8px; border-radius: 6px; box-sizing: border-box; appearance: none;" :disabled="trainingStatus === 'running'">
                            <option value="auto">Auto (智能决断)</option>
                            <option value="SGD">SGD (梯度下降)</option>
                            <option value="AdamW">AdamW (自适应增强)</option>
                        </select>
                    </div>

                    <div style="flex: 1;"></div> <!-- Spacer -->

                    <div style="margin-bottom: 16px; padding: 12px; background: rgba(59, 130, 246, 0.05); border: 1px dashed #93c5fd; border-radius: 8px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                            <span style="font-size: 12px; color: #64748b;">最新生成模型版本:</span>
                            <strong style="font-size: 12px; color: #3b82f6;">{{ modelRuns?.[0]?.name || '--' }}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="font-size: 12px; color: #64748b;">历史累计训练次数:</span>
                            <strong style="font-size: 12px; color: #3b82f6;">{{ projectStats?.run_count || 0 }} 次</strong>
                        </div>
                    </div>

                    <button v-if="trainingStatus !== 'running'" @click="startTraining" style="width: 100%; padding: 12px; background: linear-gradient(135deg, #a21caf, #6b21a8); color: white; border: none; border-radius: 8px; font-weight: bold; cursor: pointer; box-shadow: 0 4px 15px rgba(162, 28, 175, 0.4); margin-bottom: 10px;">
                        🚀 下发训练指令
                    </button>
                    <button v-else style="width: 100%; padding: 12px; background: transparent; border: 1px dashed #4ade80; color: #4ade80; border-radius: 8px; font-weight: bold; cursor: not-allowed; margin-bottom: 10px;">
                        <span class="pulse">●</span> 正在计算中...
                    </button>
                </div>

                <!-- 右侧: 图表区和日志区 -->
                <div style="flex: 3; display: flex; flex-direction: column; gap: 20px;">
                    <div style="flex: 2; display: flex; gap: 20px;">
                        <div style="flex: 1; background: white; border-radius: 12px; padding: 16px; border: 1px solid #e2e8f0; position: relative;">
                            <h4 style="font-size: 14px; color: #4a5568; margin: 0 0 10px 0; font-weight: bold;">📉 实时 Box Loss / Train Loss 衰减</h4>
                            <div id="chart-loss" style="width: 100%; height: 90%;"></div>
                        </div>
                        <div style="flex: 1; background: white; border-radius: 12px; padding: 16px; border: 1px solid #e2e8f0; position: relative;">
                            <h4 style="font-size: 14px; color: #4a5568; margin: 0 0 10px 0; font-weight: bold;">🎯 验证集 mAP@50 / Accuracy 精度</h4>
                            <div id="chart-map" style="width: 100%; height: 90%;"></div>
                        </div>
                    </div>
                    <!-- 日志输出窗 -->
                    <div style="flex: 1; background: #1e293b; border-radius: 12px; border: 1px solid #334155; display: flex; flex-direction: column; overflow: hidden; padding: 16px;">
                        <div style="font-size: 13px; color: #94a3b8; font-weight: bold; margin-bottom: 8px;">🖥️ 训练终端实时日志 (Training Logs)</div>
                        <div style="flex: 1; background: #0f172a; border-radius: 8px; padding: 12px; overflow-y: auto; font-family: monospace; font-size: 12px; color: #a5b4fc; white-space: pre-wrap; word-break: break-all;" id="trainingLogBox">
                            {{ trainingLogText || "Waiting for training sequence initiation..." }}
                        </div>
                    </div>
                </div>
            </div>

            <!-- Tab 2: Grad CAM -->
            <div v-show="sandboxTab === 'cam'" style="display: flex; flex-direction: column; flex: 1; align-items: center; justify-content: center; gap: 24px; background: #1e293b; border-radius: 12px; border: 1px solid #334155; padding: 32px 0;">
                <h3 style="color: #e2e8f0; font-size: 22px; font-weight: bold; margin: 0;">🔮 内网透视 / Grad-CAM 注意力焦点剥离测试</h3>
                <p style="color: #94a3b8; font-size: 14px; max-width: 800px; text-align: center;">请挂载一张新的测试图片。我们将贯穿当前模型的深层卷积神经网络，并通过伪彩色热力图显示大模型判定此目标时聚焦的最佳判别特征区。</p>
                <div style="display: flex; gap: 40px; align-items: stretch; width: 85%; max-width: 1200px; min-height: 450px;">
                    <div style="flex: 1; border: 2px dashed #475569; border-radius: 12px; display: flex; align-items: center; justify-content: center; position: relative; cursor: pointer; overflow: hidden; background: rgba(15, 23, 42, 0.5);" @click="triggerCamUpload">
                        <span v-if="!camInputUrl" style="color: #94a3b8; font-weight: bold; font-size: 16px;">[点击挂载] 本地检验图像</span>
                        <img v-else :src="camInputUrl" style="width: 100%; height: 100%; object-fit: contain;">
                        <input type="file" ref="camFileRef" style="display: none;" @change="handleCamUpload" accept="image/*">
                    </div>
                    <div style="color: #475569; font-size: 40px; display: flex; align-items: center;">➔</div>
                    <div style="flex: 1; border: 1px solid #334155; background: #0f172a; border-radius: 12px; display: flex; align-items: center; justify-content: center; position: relative; overflow: hidden;">
                        <span v-if="!camResultUrl && !isCamLoading" style="color: #475569; font-size: 16px;">暂无解析特征层...</span>
                        <div v-if="isCamLoading" class="pulse" style="color: #a21caf; font-weight: bold; font-size: 16px;">正在穿透推理神经网路...</div>
                        <img v-if="camResultUrl && !isCamLoading" :src="camResultUrl" style="width: 100%; height: 100%; object-fit: contain; box-shadow: 0 0 40px rgba(220, 38, 38, 0.3);">
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
                <div style="flex: 1; background: #1e293b; border-radius: 12px; border: 1px solid #334155; position: relative; overflow: hidden;">
                    <div style="position: absolute; top: 16px; right: 24px; z-index: 10;">
                        <select v-model="selectedPcaCategory" @change="updatePcaChartSeries" style="background: rgba(15, 23, 42, 0.8); color: #e2e8f0; border: 1px solid #475569; padding: 6px 12px; border-radius: 6px; font-size: 13px; outline: none; cursor: pointer;">
                            <option value="ALL">显示全部类别 (ALL)</option>
                            <option v-for="cat in pcaCategories" :key="cat" :value="cat">类别: {{ cat }}</option>
                        </select>
                    </div>
                    <div id="chart-pca" style="width: 100%; height: 100%;"></div>
                </div>
            </div>
        </div>

        <!-- Model Management Secondary Page View -->
        <div v-show="isModelManagementOpen" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: #f1f5f9; z-index: 100; display: flex; flex-direction: column; border-radius: 16px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <!-- Header -->
            <div style="background: white; border-bottom: 1px solid #e2e8f0; padding: 16px 32px; display: flex; align-items: center; justify-content: space-between;">
                <div style="display: flex; align-items: center; gap: 16px;">
                    <button @click="closeModelManagement" style="background: white; border: 1px solid #cbd5e0; border-radius: 8px; padding: 8px 16px; font-weight: bold; cursor: pointer; color: #4a5568; display: flex; align-items: center; gap: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                        ◀ 返回管线
                    </button>
                    <h3 style="font-size: 20px; font-weight: bold; margin: 0; color: #2d3748;">模型版本调度中枢</h3>
                </div>
            </div>
            
            <div style="flex: 1; padding: 24px; display: flex; flex-direction: column; gap: 24px; overflow-y: auto;">
                <!-- Runs Table -->
                <div style="background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; overflow: hidden; max-height: 40%; flex-shrink: 0;">
                    <table style="width: 100%; border-collapse: collapse; text-align: left;">
                        <thead style="background: #f8fafc; border-bottom: 2px solid #e2e8f0;">
                            <tr>
                                <th style="padding: 12px 16px; color: #4a5568; font-size: 14px; width: 60px;">序号</th>
                                <th style="padding: 12px 24px; color: #4a5568; font-size: 14px;">实验批次/模型版本</th>
                                <th style="padding: 12px 24px; color: #4a5568; font-size: 14px;">状态</th>
                                <th style="padding: 12px 24px; color: #4a5568; font-size: 14px;">构建时间</th>
                                <th style="padding: 12px 24px; color: #4a5568; font-size: 14px;">权重包</th>
                                <th style="padding: 12px 24px; color: #4a5568; font-size: 14px;">操作</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr v-for="(run, index) in modelRuns" :key="run.name" 
                                @click="selectRun(run.name)"
                                :style="{ background: selectedRunName === run.name ? '#ebf4ff' : 'white', cursor: 'pointer', borderBottom: '1px solid #e2e8f0', transition: 'background 0.2s' }">
                                <td style="padding: 12px 16px; color: #718096; font-weight: bold;">{{ index + 1 }}</td>
                                <td style="padding: 12px 24px; font-weight: bold; color: #2b6cb0;">{{ run.name }}</td>
                                <td style="padding: 12px 24px;">
                                    <span v-if="run.status === 'completed'" style="font-size: 12px; padding: 4px 8px; border-radius: 4px; background: #c6f6d5; color: #22543d; font-weight: bold;">训练完成</span>
                                    <span v-else-if="run.status === 'running'" style="font-size: 12px; padding: 4px 8px; border-radius: 4px; background: #feebc8; color: #c05621; font-weight: bold;">正在训练(后台)</span>
                                    <span v-else style="font-size: 12px; padding: 4px 8px; border-radius: 4px; background: #fed7d7; color: #822727; font-weight: bold;">失败/中断</span>
                                </td>
                                <td style="padding: 12px 24px; color: #718096; font-size: 14px;">{{ run.created_at }}</td>
                                <td style="padding: 12px 24px; color: #718096; font-size: 14px;">
                                    <span v-if="run.has_weights">best.pt, last.pt</span>
                                    <span v-else>无</span>
                                </td>
                                <td style="padding: 12px 24px;">
                                    <button @click.stop="deleteRun(run.name)" style="padding: 6px 12px; background: #fed7d7; color: #c53030; border: none; border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: bold;">移除</button>
                                </td>
                            </tr>
                            <tr v-if="modelRuns.length === 0">
                                <td colspan="6" style="padding: 24px; text-align: center; color: #a0aec0;">暂无历史训练批次</td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <!-- Selected Run Details -->
                <div v-if="selectedRunName" style="display: flex; gap: 24px; flex: 1; min-height: 400px;">
                    <div style="flex: 1; background: white; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; display: flex; flex-direction: column;">
                        <h4 style="font-size: 16px; font-weight: bold; margin: 0 0 16px 0; color: #2d3748; border-bottom: 2px solid #edf2f7; padding-bottom: 8px;">超参数台账</h4>
                        <div v-if="isRunDetailsLoading" style="color: #a0aec0; text-align: center; margin-top: 20px;">正在加载参数分布...</div>
                        <div v-else-if="activeRunDetails && activeRunDetails.args && Object.keys(activeRunDetails.args).length > 0" style="overflow-y: auto; flex: 1; font-size: 13px; color: #4a5568; line-height: 1.8;">
                            <div v-for="(val, key) in activeRunDetails.args" :key="key" style="display: flex; justify-content: space-between; border-bottom: 1px dashed #e2e8f0; padding: 4px 0;">
                                <span style="font-weight: bold; color: #4a5568;">{{ key }}</span>
                                <span style="color: #718096;">{{ val }}</span>
                            </div>
                        </div>
                        <div v-else style="color: #a0aec0; text-align: center; margin-top: 20px;">暂无参数信息</div>
                    </div>

                    <div style="flex: 2; background: white; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; display: flex; flex-direction: column; gap: 20px;">
                        <div style="flex: 1; position: relative; border: 1px solid #e2e8f0; border-radius: 8px;">
                            <h4 style="font-size: 14px; font-weight: bold; margin: 10px; color: #2d3748; position: absolute; z-index: 10;">📈 验证集精度 (mAP@50 / Top1-Acc)</h4>
                            <div id="manage-chart-map" style="width: 100%; height: 100%;"></div>
                        </div>
                        <div style="flex: 1; position: relative; border: 1px solid #e2e8f0; border-radius: 8px;">
                            <h4 style="font-size: 14px; font-weight: bold; margin: 10px; color: #2d3748; position: absolute; z-index: 10;">📉 模型训练损失 (Train Loss)</h4>
                            <div id="manage-chart-loss" style="width: 100%; height: 100%;"></div>
                        </div>
                    </div>
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
                border: 1px solid #cbd5e0;
                color: #4a5568;
                padding: 6px 14px;
                border-radius: 20px;
                font-size: 13px;
                cursor: pointer;
                transition: all 0.2s;
            }
            .sandbox-nav.active {
                background: #fdf4ff;
                border-color: #d6bcfa;
                color: #805ad5;
                font-weight: bold;
            }
            .sandbox-nav:hover:not(.active) {
                background: #edf2f7;
                color: #2d3748;
            }
        </style>
    </div>
    `,
    setup() {
        const projects = ref([]);

        const fetchProjects = async () => {
            try {
                const url = window.location.origin.includes('5173') ? 'http://127.0.0.1:8000/api/model_project/list' : '/api/model_project/list';
                const res = await fetch(url);
                if (res.ok) {
                    const json = await res.json();
                    if (json.status === 'success') {
                        projects.value = json.data.map((d, idx) => ({
                            id: idx + 1,
                            name: d.name,
                            desc: d.description || '新建工程。',
                            updatedAt: new Date().toISOString().split('T')[0]
                        }));
                    }
                }
            } catch (e) { console.error(e) }
        };

        onMounted(() => {
            fetchProjects();
        });

        const activeProject = ref(null);
        const showCreateModal = ref(false);
        const newProjectForm = ref({ name: '', desc: '' });

        // Custom Toast System
        const toastMessage = ref('');
        const toastType = ref('success');
        let toastTimeout = null;

        const showToast = (msg, type = 'success') => {
            toastMessage.value = msg;
            toastType.value = type;
            if (toastTimeout) clearTimeout(toastTimeout);
            toastTimeout = setTimeout(() => { toastMessage.value = ''; }, 3000);
        };

        // Phase 1 Dropdowns fetching pure Datasets
        const datasetList = ref([]);
        const fetchDatasets = async () => {
            try {
                const url = window.location.origin.includes('5173') ? 'http://127.0.0.1:8000/api/dataset/list' : '/api/dataset/list';
                const res = await fetch(url);
                if (res.ok) {
                    const json = await res.json();
                    if (json.status === 'success') {
                        datasetList.value = json.data.map(d => ({ name: d.name, type: 'Data' }));
                    }
                }
            } catch (e) { console.error(e) }
        };

        const selectedDataset = ref('');
        const selectedDataVersion = ref('');
        const availableDataVersions = ref([]);
        const selectedModel = ref('');

        watch(selectedDataset, async (newVal) => {
            selectedDataVersion.value = '';
            availableDataVersions.value = [];
            if (!newVal) return;
            try {
                const url = window.location.origin.includes('5173')
                    ? `http://127.0.0.1:8000/api/dataset/${newVal}/versions`
                    : `/api/dataset/${newVal}/versions`;
                const res = await fetch(url);
                if (res.ok) {
                    const json = await res.json();
                    if (json.status === 'success' && json.data && json.data.length > 0) {
                        availableDataVersions.value = json.data;
                        selectedDataVersion.value = json.data[0].version; // auto-select first
                    } else {
                        // fallback if no versions created yet, just mock v1
                        availableDataVersions.value = [{ version: 'v1', description: '基础标注' }];
                        selectedDataVersion.value = 'v1';
                    }
                }
            } catch (e) { console.error(e) }
        });

        const openProject = (proj) => {
            activeProject.value = proj;
            fetchDatasets(); // Fetch pure datasets when entering a project pipeline
            isSandboxOpen.value = false;
            isModelManagementOpen.value = false;
        };

        const createProject = async () => {
            if (!newProjectForm.value.name.trim()) return;
            try {
                const url = window.location.origin.includes('5173') ? 'http://127.0.0.1:8000/api/model_project/create' : '/api/model_project/create';
                const res = await fetch(url, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ project_name: newProjectForm.value.name.trim(), description: newProjectForm.value.desc })
                });
                if (res.ok) {
                    fetchProjects();
                    showCreateModal.value = false;
                    newProjectForm.value = { name: '', desc: '' };
                } else {
                    const err = await res.json();
                    showToast(err.detail || '创建失败', 'error');
                }
            } catch (e) { console.error(e) }
        };

        const showRenameModal = ref(false);
        const renameForm = ref({ name: '', proj: null });

        const showDeleteModal = ref(false);
        const deleteTarget = ref(null);

        const deleteProject = (proj) => {
            deleteTarget.value = proj;
            showDeleteModal.value = true;
        };

        const confirmDeleteProject = async () => {
            const proj = deleteTarget.value;
            if (!proj) return;
            try {
                const url = window.location.origin.includes('5173')
                    ? `http://127.0.0.1:8000/api/model_project/${proj.name}` : `/api/model_project/${proj.name}`;
                const res = await fetch(url, { method: 'DELETE' });
                if (res.ok) {
                    fetchProjects();
                    if (activeProject.value && activeProject.value.name === proj.name) activeProject.value = null;
                    showDeleteModal.value = false;
                    showToast(`项目 ${proj.name} 已成功删除`);
                } else {
                    const err = await res.json();
                    console.error('删除失败', err);
                    showToast('删除项目失败', 'error');
                }
            } catch (e) { console.error(e) }
        };

        const renameProject = (proj) => {
            renameForm.value = { name: proj.name, proj: proj };
            showRenameModal.value = true;
        };

        const confirmRenameProject = async () => {
            const proj = renameForm.value.proj;
            const newName = renameForm.value.name.trim();
            if (!newName || newName === proj.name) {
                showRenameModal.value = false;
                return;
            }
            try {
                const url = window.location.origin.includes('5173')
                    ? `http://127.0.0.1:8000/api/model_project/rename/${proj.name}` : `/api/model_project/rename/${proj.name}`;
                const res = await fetch(url, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ new_name: newName })
                });
                if (res.ok) {
                    fetchProjects();
                    if (activeProject.value && activeProject.value.name === proj.name) activeProject.value.name = newName;
                    showRenameModal.value = false;
                } else {
                    const err = await res.json();
                    console.error('重命名失败', err);
                }
            } catch (e) { console.error(e) }
        };
        const isSandboxOpen = ref(false);
        const sandboxTab = ref('training');
        const trainingStatus = ref('idle');
        const trainConfig = ref({ epochs: 10, batch: 8, imgsz: 224, optimizer: 'auto', lr0: 0.01, patience: 50 });
        const trainingLogText = ref('');
        const projectStats = ref({ run_count: 0 });

        let lossChart = null;
        let mapChart = null;
        let pcaChart = null;
        let lossData = [];
        let mapData = [];
        let curWebSocket = null;
        let pollingInterval = null;

        // This holds the selected run name for testing (Grad-CAM, PCA)
        const selectedRunForTesting = ref('');

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
                xAxis: { type: 'category', data: [], axisLine: { lineStyle: { color: '#cbd5e0' } }, axisLabel: { color: '#4a5568' } },
                yAxis: { type: 'value', splitLine: { lineStyle: { color: '#e2e8f0' } }, axisLabel: { color: '#4a5568' } },
                tooltip: { trigger: 'axis', backgroundColor: '#ffffff', borderColor: '#e2e8f0', textStyle: { color: '#2d3748' } }
            };

            lossChart.setOption({
                ...commonOptions,
                series: [{ name: 'Box Loss / Train Loss', type: 'line', data: [], smooth: true, lineStyle: { color: '#e53e3e', width: 3 }, showSymbol: false, itemStyle: { color: '#e53e3e' } }]
            });

            mapChart.setOption({
                ...commonOptions,
                series: [{ name: 'Val mAP@50 / Accuracy', type: 'line', data: [], smooth: true, lineStyle: { color: '#3182ce', width: 3 }, showSymbol: false, itemStyle: { color: '#3182ce' }, areaStyle: { color: new window.echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: 'rgba(49,130,206,0.3)' }, { offset: 1, color: 'rgba(49,130,206,0)' }]) } }]
            });
        };

        const pcaCategories = ref([]);
        const selectedPcaCategory = ref('ALL');
        let fullPcaPoints = [];

        const initPcaChart = (points) => {
            if (!pcaChart) {
                pcaChart = window.echarts.init(document.getElementById('chart-pca'));
            }
            fullPcaPoints = points;

            const seriesData = {};
            points.forEach(p => {
                if (!seriesData[p.label]) seriesData[p.label] = [];
                seriesData[p.label].push([p.x, p.y]);
            });

            pcaCategories.value = Object.keys(seriesData);

            updatePcaChartSeries();
        };

        const updatePcaChartSeries = () => {
            if (!pcaChart) return;
            const seriesData = {};
            // Only include the selected category or all
            fullPcaPoints.forEach(p => {
                if (selectedPcaCategory.value === 'ALL' || p.label === selectedPcaCategory.value) {
                    if (!seriesData[p.label]) seriesData[p.label] = [];
                    seriesData[p.label].push([p.x, p.y]);
                }
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
                legend: { show: false }, // Hide Echarts legend as custom dropdown is used
                xAxis: { type: 'value', splitLine: { show: false }, axisLine: { lineStyle: { color: '#475569' } } },
                yAxis: { type: 'value', splitLine: { lineStyle: { color: '#334155', type: 'dashed' } }, axisLine: { lineStyle: { color: '#475569' } } },
                series: series,
                color: ['#4ade80', '#fbbf24', '#f87171', '#c084fc', '#60a5fa', '#f472b6', '#2dd4bf', '#a3e635']
            }, true); // use true to merge strictly
        };

        const enterSandbox = async (targetTab = 'training') => {
            if (!selectedDataset.value || !selectedDataVersion.value || !selectedModel.value) return;
            sandboxTab.value = targetTab;
            isSandboxOpen.value = true;
            nextTick(() => {
                initCharts();
            });

            // 1. Ensure Model Runs list is loaded so the "Latest version" UI gets populated
            await fetchRuns();

            // 2. Fetch real historical project stats
            try {
                const url = window.location.origin.includes('5173')
                    ? `http://127.0.0.1:8000/api/training/stats/${activeProject.value.name}`
                    : `/api/training/stats/${activeProject.value.name}`;
                const res = await fetch(url);
                if (res.ok) {
                    const json = await res.json();
                    if (json.status === 'success') {
                        projectStats.value = json.data || { run_count: 0 };
                    } else {
                        projectStats.value = { run_count: 0 };
                    }
                } else {
                    projectStats.value = { run_count: 0 };
                }
            } catch (e) {
                console.error("Failed to fetch project stats", e);
            }

            // 3. Recover active training state if the backend is actively running an epoch
            try {
                const url = window.location.origin.includes('5173')
                    ? `http://127.0.0.1:8000/api/training/classify/active?project_name=${activeProject.value.name}`
                    : `/api/training/classify/active?project_name=${activeProject.value.name}`;
                const res = await fetch(url);
                if (res.ok) {
                    const json = await res.json();
                    if (json.is_running && json.active_run) {
                        startPollingTrainingLogs(json.active_run);
                    }
                }
            } catch (e) {
                console.error("Failed to query active training state", e);
            }
        };

        const exitSandbox = () => {
            isSandboxOpen.value = false;
            if (pollingInterval) clearInterval(pollingInterval);
        };

        const startTraining = async () => {
            trainingStatus.value = 'running';
            trainingLogText.value = 'Initiating training sequence...';
            lossData = []; mapData = [];
            const emptyOption = { xAxis: { data: [] }, series: [{ data: [] }] };
            lossChart.setOption(emptyOption);
            mapChart.setOption(emptyOption);

            if (pollingInterval) clearInterval(pollingInterval);

            const jobId = "run_" + new Date().getTime();

            // POST Request to correct endpoint
            try {
                const url = window.location.origin.includes('5173') ? 'http://127.0.0.1:8000/api/training/classify/start' : '/api/training/classify/start';
                const payload = {
                    project_name: activeProject.value.name,
                    dataset_name: selectedDataset.value,
                    version: selectedDataVersion.value,
                    run_name: jobId,
                    model: selectedModel.value,
                    epochs: trainConfig.value.epochs,
                    batch_size: trainConfig.value.batch,
                    imgsz: trainConfig.value.imgsz,
                    lr0: trainConfig.value.lr0,
                    patience: trainConfig.value.patience
                };

                const response = await fetch(url, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) {
                    const err = await response.json();
                    console.error("Training start failed:", err);
                    showToast(err.detail || "启动失败，请检查配置或后端状态。", "error");
                    trainingStatus.value = 'idle';
                    return;
                }

                console.log("[Launch] Command sent tracking job", jobId);

                // Start polling mechanism since backend has no websocket for classify
                startPollingTrainingLogs(jobId);

            } catch (err) {
                console.error("Start training network error:", err);
                showToast(`启动错误: ${err.message || '网络或接口请求错误。'}`, "error");
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
            if (!selectedRunForTesting.value) {
                alert("请先在顶部下拉菜单选择一个已训练的模型版本");
                return;
            }
            isCamLoading.value = true;
            try {
                const formData = new FormData();
                formData.append('file', camFileRaw.value);
                const res = await axios.post(`/api/analytica/grad_cam?project_name=${activeProject.value.name}&run_name=${selectedRunForTesting.value}`, formData, {
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
            if (!selectedRunForTesting.value) {
                alert("请先在顶部下拉菜单选择一个已训练的模型版本");
                return;
            }
            isPcaLoading.value = true;
            selectedPcaCategory.value = 'ALL';
            try {
                const res = await axios.post(`/api/analytica/pca_cluster`, { project_name: activeProject.value.name, run_name: selectedRunForTesting.value });
                if (res.data.status === 'success') {
                    initPcaChart(res.data.points);
                }
            } catch (e) {
                console.error(e);
            } finally {
                isPcaLoading.value = false;
            }
        };

        // Model Management Logic
        const isModelManagementOpen = ref(false);
        const modelRuns = ref([]);
        const selectedRunName = ref('');
        const activeRunDetails = ref(null);
        const isRunDetailsLoading = ref(false);

        let manageAccChart = null;
        let manageLossChart = null;

        const openModelManagement = () => {
            isModelManagementOpen.value = true;
            fetchRuns();
            activeRunDetails.value = null;
            selectedRunName.value = '';
        };

        const closeModelManagement = () => {
            isModelManagementOpen.value = false;
            activeRunDetails.value = null;
            selectedRunName.value = '';
        };

        async function fetchRuns() {
            try {
                const url = window.location.origin.includes('5173')
                    ? `http://127.0.0.1:8000/api/training/runs?project_name=${activeProject.value.name}`
                    : `/api/training/runs?project_name=${activeProject.value.name}`;
                const res = await fetch(url);
                if (res.ok) {
                    const json = await res.json();
                    if (json.status === 'success') {
                        modelRuns.value = json.data.map(run => ({
                            name: run.run_name,
                            status: run.status,
                            created_at: run.created_at,
                            has_weights: run.has_weights
                        }));
                        // Auto-select latest run if none is selected
                        if (modelRuns.value.length > 0 && !selectedRunForTesting.value) {
                            selectedRunForTesting.value = modelRuns.value[0].name;
                        }
                    }
                }
            } catch (err) {
                console.error(err);
            }
        }

        const selectRun = (name) => {
            selectedRunName.value = name;
        };

        const deleteRun = async (name) => {
            if (!confirm(`确定要移除批次 ${name} 吗？这将会删除相关的模型权重和训练记录。`)) return;
            try {
                const url = window.location.origin.includes('5173')
                    ? `http://127.0.0.1:8000/api/training/runs/${name}?project_name=${activeProject.value.name}`
                    : `/api/training/runs/${name}?project_name=${activeProject.value.name}`;
                const res = await fetch(url, { method: 'DELETE' });
                if (res.ok) {
                    modelRuns.value = modelRuns.value.filter(r => r.name !== name);
                    if (selectedRunName.value === name) {
                        selectedRunName.value = '';
                        activeRunDetails.value = null;
                    }
                } else {
                    alert('移除失败');
                }
            } catch (err) {
                console.error(err);
                alert('网络错误');
            }
        };

        const fetchRunDetails = async (runName) => {
            if (!runName) return;
            isRunDetailsLoading.value = true;
            try {
                const url = window.location.origin.includes('5173')
                    ? `http://127.0.0.1:8000/api/training/runs/${runName}/details?project_name=${activeProject.value.name}`
                    : `/api/training/runs/${runName}/details?project_name=${activeProject.value.name}`;
                const res = await fetch(url);
                if (res.ok) {
                    const json = await res.json();
                    if (json.status === 'success') {
                        activeRunDetails.value = json.data;
                        renderManageCharts(activeRunDetails.value.results);
                    } else {
                        activeRunDetails.value = { args: {}, results: [] };
                        renderManageCharts([]);
                    }
                } else {
                    activeRunDetails.value = { args: {}, results: [] };
                    renderManageCharts([]);
                }
            } catch (e) {
                console.error(e);
            } finally {
                isRunDetailsLoading.value = false;
            }
        };

        watch(selectedRunName, (newVal) => {
            if (newVal && isModelManagementOpen.value) {
                fetchRunDetails(newVal);
            }
        });

        const renderManageCharts = (results) => {
            nextTick(() => {
                if (!window.echarts) return;
                let domMap = document.getElementById('manage-chart-map');
                let domLoss = document.getElementById('manage-chart-loss');
                if (!domMap || !domLoss) return;

                if (!manageAccChart) manageAccChart = window.echarts.init(domMap);
                if (!manageLossChart) manageLossChart = window.echarts.init(domLoss);

                manageAccChart.clear();
                manageLossChart.clear();

                const epochs = [];
                const maps = [];
                const boxloss = [];
                results.forEach(r => {
                    epochs.push(`Ep ${r.epoch}`);

                    const acc = r['metrics/accuracy_top1'] || r['metrics/mAP50(B)'] || r.metrics_mAP50 || 0;
                    maps.push(parseFloat(acc));

                    const loss = r['train/loss'] || r['train/box_loss'] || r.train_box_loss || 0;
                    boxloss.push(parseFloat(loss));
                });

                manageAccChart.setOption({
                    grid: { left: 40, right: 20, top: 40, bottom: 20 },
                    xAxis: { type: 'category', data: epochs },
                    yAxis: { type: 'value' },
                    tooltip: { trigger: 'axis' },
                    series: [{ data: maps, type: 'line', smooth: true, lineStyle: { color: '#3182ce', width: 3 }, itemStyle: { color: '#3182ce' } }]
                });

                manageLossChart.setOption({
                    grid: { left: 40, right: 20, top: 40, bottom: 20 },
                    xAxis: { type: 'category', data: epochs },
                    yAxis: { type: 'value' },
                    tooltip: { trigger: 'axis' },
                    series: [{ data: boxloss, type: 'line', smooth: true, lineStyle: { color: '#e53e3e', width: 3 }, itemStyle: { color: '#e53e3e' } }]
                });
            });
        };

        return {
            projects,
            fetchProjects,
            datasetList,
            selectedDataVersion,
            availableDataVersions,
            activeProject,
            showCreateModal,
            newProjectForm,
            openProject,
            createProject,
            renameProject,
            deleteProject,
            showRenameModal,
            renameForm,
            confirmRenameProject,
            showDeleteModal,
            deleteTarget,
            confirmDeleteProject,
            toastMessage,
            toastType,
            selectedDataset,
            selectedModel,
            isSandboxOpen,
            sandboxTab,
            changeSandboxTab,
            enterSandbox,
            exitSandbox,
            trainingStatus,
            trainConfig,
            trainingLogText,
            startTraining,
            exportOnnx,
            camFileRef, camInputUrl, camResultUrl, isCamLoading, triggerCamUpload, handleCamUpload, executeCam,
            isPcaLoading, executePCA,
            selectedRunForTesting,
            pcaCategories, selectedPcaCategory, updatePcaChartSeries,
            isModelManagementOpen, modelRuns, selectedRunName, activeRunDetails, isRunDetailsLoading,
            openModelManagement, closeModelManagement, selectRun, deleteRun
        };
    }
};
