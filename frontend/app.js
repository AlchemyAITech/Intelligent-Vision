import { createApp, ref, shallowRef, computed, watch } from 'vue';
window.Vue = { ref, shallowRef, computed, watch }; // 供组件内部结构取用
import HomePortal from './components/HomePortal.js?v=2026.115';
import ExperienceModule from './components/ExperienceModule.js?v=2026.115';
import TrainingPlatform from './components/TrainingPlatform.js?v=2026.115';

// 用于将独立模块下沉传递（为了解耦，避免全部导入打包在一个文件内）
import SAMLab from './components/SAMLab.js?v=2026.115';
import ColorLab from './components/ColorLab.js?v=2026.115';
import ConvolutionLab from './components/ConvolutionLab.js?v=2026.115';
import CNNLab from './components/CNNLab.js?v=2026.115';
import YOLOLab from './components/YOLOLab.js?v=2026.115';
import FaceLab from './components/FaceLab.js?v=2026.115';

const LAB_REGISTRY = {
    ColorLab, ConvolutionLab, CNNLab, YOLOLab, FaceLab, SAMLab
};

const app = createApp({
    components: {
        HomePortal,
        ExperienceModule,
        TrainingPlatform
    },
    template: `
    <div id="app" style="width: 100vw; height: 100vh; overflow: hidden; position: relative;">
        <transition name="fade" mode="out-in">
            <component 
                :is="currentViewComponent" 
                :registered-labs="LAB_REGISTRY"
                @navigate="handleNavigate"
                @back-home="goHome"
            ></component>
        </transition>
    </div>
    `,
    setup() {
        // 全局第一级路由: 'home' | 'experiments' | 'training'
        const currentView = ref('home');

        const currentViewComponent = computed(() => {
            switch (currentView.value) {
                case 'home': return 'HomePortal';
                case 'experiments': return 'ExperienceModule';
                case 'training': return 'TrainingPlatform';
                default: return 'HomePortal';
            }
        });

        const handleNavigate = (dest) => {
            currentView.value = dest;
        };

        const goHome = () => {
            currentView.value = 'home';
        };

        return {
            currentView,
            currentViewComponent,
            LAB_REGISTRY,
            handleNavigate,
            goHome
        };
    }
});

app.mount('#app');
