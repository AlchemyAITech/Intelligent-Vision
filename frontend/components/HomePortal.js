import { ref, onMounted, onUnmounted } from 'vue';

export default {
    name: 'HomePortal',
    template: `
    <div class="portal-container">
        <!-- 动态粒子/流光背景 -->
        <canvas ref="bgCanvas" class="portal-bg"></canvas>
        
        <!-- 主视觉内容区 -->
        <div class="portal-content">
            <div class="portal-header">
                <div class="tsinghua-badge">THU</div>
                <h1 class="portal-title">智能视界</h1>
                <h2 class="portal-subtitle">人工智能驱动的医学图像分析</h2>
                <p class="portal-desc">从零样本探索到全链路可视化训练平台，解锁医学 AI 前沿开发能力</p>
            </div>
            
            <div class="portal-cards">
                <!-- 模块 1：小试牛刀 -->
                <div class="nav-card glass-panel" @click="navigate('experiments')">
                    <div class="card-icon">🧪</div>
                    <h3>小试牛刀</h3>
                    <p>探索零样本检测与分割，免训练体验 SAM/YOLO 极速分析工具舱。</p>
                    <div class="card-action">进入实验室 ➜</div>
                </div>

                <!-- 模块 2：训练平台 -->
                <div class="nav-card glass-panel highlight" @click="navigate('training')">
                    <div class="card-badge">核心基建</div>
                    <div class="card-icon">🚀</div>
                    <h3>训练平台</h3>
                    <p>基于 Ultralytics 的多模态连线训练平台。从大模型挂载到参数可解释分析。</p>
                    <div class="card-action">启动炼丹炉 ➜</div>
                </div>

                <!-- 模块 3：实战项目 -->
                <div class="nav-card glass-panel disabled" @click="navigate('projects')">
                    <div class="card-icon">🌟</div>
                    <h3>实战项目</h3>
                    <p>医疗图像领域的综合性实战展示大屏，降维评估与全栈流媒体测试。</p>
                    <div class="card-action">建设中...</div>
                </div>
            </div>
        </div>
        
        <div class="portal-footer">
            &copy; 2026 清华大学 · 医学图像人工智能通识课专属教具
        </div>
    </div>
    `,
    emits: ['navigate'],
    setup(props, { emit }) {
        const bgCanvas = ref(null);
        let animationFrameId = null;

        const navigate = (destination) => {
            if (destination === 'projects') return; // temporarily disabled
            emit('navigate', destination);
        };

        const initBackground = () => {
            const canvas = bgCanvas.value;
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            let width, height;

            const resize = () => {
                width = canvas.width = window.innerWidth;
                height = canvas.height = window.innerHeight;
            };
            window.addEventListener('resize', resize);
            resize();

            const particles = [];
            for (let i = 0; i < 80; i++) {
                particles.push({
                    x: Math.random() * width,
                    y: Math.random() * height,
                    radius: Math.random() * 2 + 1,
                    vx: Math.random() * 0.5 - 0.25,
                    vy: Math.random() * 0.5 - 0.25,
                    life: Math.random()
                });
            }

            const draw = () => {
                // 清华紫荆花紫主色调混合深空渐变
                const gradient = ctx.createLinearGradient(0, 0, width, height);
                gradient.addColorStop(0, '#4a1158'); // 深暗紫
                gradient.addColorStop(0.5, '#2e0a3b'); // 深紫
                gradient.addColorStop(1, '#0f0518'); // 近黑紫

                ctx.fillStyle = gradient;
                ctx.fillRect(0, 0, width, height);

                // 绘制连接线
                ctx.lineWidth = 0.5;
                for (let i = 0; i < particles.length; i++) {
                    for (let j = i + 1; j < particles.length; j++) {
                        const dx = particles[i].x - particles[j].x;
                        const dy = particles[i].y - particles[j].y;
                        const dist = Math.sqrt(dx * dx + dy * dy);
                        if (dist < 150) {
                            ctx.strokeStyle = `rgba(232, 121, 249, ${1 - dist / 150})`;
                            ctx.beginPath();
                            ctx.moveTo(particles[i].x, particles[i].y);
                            ctx.lineTo(particles[j].x, particles[j].y);
                            ctx.stroke();
                        }
                    }
                }

                // 绘制粒子
                particles.forEach(p => {
                    p.x += p.vx;
                    p.y += p.vy;
                    if (p.x < 0 || p.x > width) p.vx *= -1;
                    if (p.y < 0 || p.y > height) p.vy *= -1;

                    ctx.beginPath();
                    ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
                    ctx.fillStyle = `rgba(162, 28, 175, ${0.5 + 0.5 * Math.sin(p.life * Math.PI)})`;
                    ctx.fill();
                    p.life += 0.01;
                    if (p.life > 2) p.life = 0;
                });

                animationFrameId = requestAnimationFrame(draw);
            };
            draw();

            onUnmounted(() => {
                window.removeEventListener('resize', resize);
                if (animationFrameId) cancelAnimationFrame(animationFrameId);
            });
        };

        onMounted(() => {
            initBackground();
        });

        return {
            bgCanvas,
            navigate
        };
    }
};
