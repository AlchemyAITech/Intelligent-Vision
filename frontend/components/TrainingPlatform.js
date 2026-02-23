import { ref } from 'vue';

export default {
    name: 'TrainingPlatform',
    template: `
    <div style="display: flex; width: 100vw; height: 100vh;">
        <!-- Training Platform Sidebar -->
        <div class="sidebar" style="background: rgba(250, 245, 255, 0.95); border-right: 1px solid rgba(162, 28, 175, 0.2);">
            <div class="sidebar-header" style="cursor: pointer;" @click="$emit('back-home')" title="返回主页">
                <div style="line-height:1.2; padding-top:10px;">
                    <span style="color:#82318E; font-size:1.1em; font-weight:800;">训练平台</span><br>
                    <span style="font-size:0.75em; color:#a21caf; font-weight:600;">大模型算法工厂</span>
                </div>
            </div>
            
            <ul class="nav-list" style="margin-top:20px;">
                <li :class="['nav-item', { active: currentSection === 'data' }]" @click="currentSection = 'data'">
                    <span class="nav-icon">📂</span> <span class="nav-text">数据管理</span>
                </li>
                <li :class="['nav-item', { active: currentSection === 'projects' }]" @click="currentSection = 'projects'">
                    <span class="nav-icon">📊</span> <span class="nav-text">项目管理</span>
                </li>
            </ul>
        </div>

        <!-- Training Sandbox Area -->
        <div class="main-content" style="background: #fafafc;">
            <div v-if="currentSection === 'data'">
                <h2>数据管理 (即将开放)</h2>
                <p>支持批量图像导入，支持自动与半自动（SAM零样本辅助）多模态标注闭环区。支持 COCO 与 YOLO 格式。</p>
            </div>
            <div v-else-if="currentSection === 'projects'">
                <h2>模型项目工程仓 (即将开放)</h2>
                <p>基于硬件极速器 (MPS/CUDA) 的 Ultralytics 并行图谱。在此组装链路、拉起 Zero/Few Shot 及挂载验证集降维分析阵列。</p>
            </div>
        </div>
    </div>
    `,
    emits: ['back-home'],
    setup() {
        const currentSection = ref('projects');
        return {
            currentSection
        };
    }
};
