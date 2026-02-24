import { ref, computed } from 'vue';

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
        <div v-else style="flex: 1; display: flex; flex-direction: column;">
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

        const enterSandbox = () => {
            if (!selectedDataset.value || !selectedModel.value) {
                alert('请先完整挂载数据流与算法底座。');
                return;
            }
            alert('即将切换至 Phase 3/4 的训练监控沙盘...');
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
            enterSandbox
        };
    }
};
