import { ref } from 'vue';
import DataManagement from './DataManagement.js?v=2026.115';
import ProjectManagement from './ProjectManagement.js?v=2026.115';

export default {
    name: 'TrainingPlatform',
    components: {
        DataManagement,
        ProjectManagement
    },
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
        <div class="main-content" style="background: #fafafc; flex: 1; overflow: auto;">
            <DataManagement v-if="currentSection === 'data'" />
            <ProjectManagement v-else-if="currentSection === 'projects'" />
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
