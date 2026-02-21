import { createApp, ref, shallowRef, computed } from 'vue';
import SAMLab from './components/SAMLab.js?v=2026.55';
import ColorLab from './components/ColorLab.js?v=2026.55';
import ConvolutionLab from './components/ConvolutionLab.js?v=2026.55';
import CNNLab from './components/CNNLab.js?v=2026.55';
import YOLOLab from './components/YOLOLab.js?v=2026.55';
import FaceLab from './components/FaceLab.js?v=2026.55';

const app = createApp({
    template: `
    <div id="app" style="display: flex; width: 100vw; height: 100vh;">
        <!-- Sidebar Navigation -->
        <div class="sidebar" :class="{ collapsed: isCollapsed }">
            <div class="sidebar-header">
                <div v-show="!isCollapsed" style="line-height:1.2;">
                    <span>智能视界</span><br>
                    <span style="font-size:0.7em; opacity:0.8;">Intelligent Vision</span>
                </div>
                <span v-show="isCollapsed">🔬</span>
            </div>
            <ul class="nav-list">
                <li v-for="tab in tabs" :key="tab.id" :class="['nav-item', { active: currentTab === tab.id }]"
                    @click="currentTab = tab.id" :title="tab.name">
                    <span class="nav-icon">{{ tab.icon }}</span>
                    <span v-show="!isCollapsed" class="nav-text">{{ tab.name }}</span>
                </li>
            </ul>
            <div class="sidebar-copyright" v-show="!isCollapsed">
                Tsinghua University<br>General Education Course 2026
            </div>
            <div class="collapse-btn" @click="toggleSidebar">
                <span v-if="!isCollapsed">◀ 收起栏目</span>
                <span v-else>▶</span>
            </div>
        </div>

        <!-- Main Content Area -->
        <div class="main-content">
            <!-- Dynamic Component Rendering -->
            <transition name="fade" mode="out-in">
                <component :is="currentComponent"></component>
            </transition>
        </div>
    </div>
    `,
    setup() {
        const tabs = ref([
            { id: 'ColorLab', name: '图像的本质', icon: '🎨', component: ColorLab },
            { id: 'ConvolutionLab', name: '卷积实验室', icon: '⚙️', component: ConvolutionLab },
            { id: 'CNNLab', name: '神经网络实验室', icon: '🧠', component: CNNLab },
            { id: 'YOLOLab', name: 'YOLO实验室', icon: '👁️', component: YOLOLab },
            { id: 'FaceLab', name: '人脸实验室', icon: '👤', component: FaceLab },
            { id: 'SAMLab', name: 'SAM实验室', icon: '✨', component: SAMLab }
        ]);

        const currentTab = ref('ColorLab');
        const isCollapsed = ref(false);

        const toggleSidebar = () => {
            isCollapsed.value = !isCollapsed.value;
        };

        const currentComponent = computed(() => {
            const tab = tabs.value.find(t => t.id === currentTab.value);
            return tab ? tab.component : null;
        });

        return {
            tabs,
            currentTab,
            isCollapsed,
            toggleSidebar,
            currentComponent
        };
    }
});

app.mount('#app');
