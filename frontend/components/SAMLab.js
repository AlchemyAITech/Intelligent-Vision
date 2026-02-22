import { ref, onMounted, onUnmounted, nextTick, computed, watch } from 'vue';
import ImageSource from './ImageSource.js';

export default {
    name: 'SAMLab',
    components: {
        ImageSource
    },
    template: `
    <div class="sam-lab-unified fullscreen-mode">
        
        <!-- Workspace (Full Screen Background Area) -->
        <div class="workspace-area" @wheel.prevent="handleWheel"
             @contextmenu.prevent
             style="position: absolute; top:0; left:0; right:0; bottom:0; overflow:hidden;"
             @mousedown="globalSpacePushed ? startPan($event) : null"
             @mousemove="globalSpacePushed ? doPan($event) : null"
             @mouseup="endPan"
             @mouseleave="endPan">
             
            <!-- 将特征加载状态融合到操作提示块内 -->
            <div v-if="sessionId" class="status-indicator mini-status" :class="{ loading: isLoading }" style="margin-right: 15px; border-right: 1px solid #ddd; padding-right: 15px;">
                <div class="pulse-dot"></div>
                <span style="font-size: 12px; font-weight: bold; color: #a21caf;">{{ statusMessage }}</span>
            </div>

            <!-- 增加 video 和原本的 img 在同一个容器并保持宽高同步 -->
            <div class="canvas-container"
                 :style="{ transform: 'translate(' + panOffset.x + 'px, ' + panOffset.y + 'px) scale(' + zoomLevel + ')' }"
                 @mousedown="!globalSpacePushed ? handleMouseDown($event) : null"
                 @mousemove="!globalSpacePushed ? handleMouseMove($event) : null"
                 @mouseup="!globalSpacePushed ? handleMouseUp($event) : null"
                 @mouseleave="!globalSpacePushed ? isDragging=false : null">
                 
                 <video v-if="videoUrl && subTab === 'tracking'" :src="videoUrl" ref="videoRef"
                        style="display: block; width: 100%; height: auto; border-radius: 8px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); pointer-events: none;"
                        @loadedmetadata="handleVideoLoaded">
                 </video>
                 
                <img v-else-if="imageUrl" :src="imageUrl" ref="imageRef" @load="onImageLoaded"
                     draggable="false"
                     style="display: block; width: 100%; height: auto; border-radius: 8px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);" />
                
                <canvas ref="canvasRef"
                        style="position: absolute; top:0; left:0; width: 100%; height: 100%; pointer-events: none; z-index: 5;">
                </canvas>
            </div>
        </div>


        <style>
            @keyframes spin {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
            }
        </style>
        <!-- 全局加载遮罩 (Loading Overlay) -->
        <div v-if="isLoading" class="loading-overlay" 
             style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.4); backdrop-filter: blur(4px); z-index: 10000; display: flex; flex-direction: column; align-items: center; justify-content: center; color: white;">
            <div class="loader-spinner" style="width: 50px; height: 50px; border: 5px solid rgba(255,255,255,0.3); border-top: 5px solid #fff; border-radius: 50%; animation: spin 1s linear infinite;"></div>
            <p style="margin-top: 15px; font-weight: 500; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">{{ statusMessage }}</p>
        </div>
        <div class="top-floating-bar" style="z-index: 1000;">
            <h2 class="app-title">✨ SAM实验室</h2>
            
            <div class="tab-pill-group">
                <div v-for="t in ['labeling', 'tracking', 'recognition']" :key="t"
                     class="tab-pill"
                     :class="{ active: subTab === t }"
                     @click="subTab = t">
                    {{ t === 'labeling' ? '交互式标注' : (t === 'tracking' ? '零样本追踪' : '零样本识别') }}
                </div>
            </div>
        </div>

        <!-- 左侧功能悬浮区 (Z-Index: 1000) -->
        <div class="left-floating-panel" style="z-index: 1000;">
            <!-- (2) 图像/视频加载 -->
            <div class="glass-card">
                <button class="btn-primary full-btn" @click="showUploadModal = true">
                    {{ subTab === 'tracking' ? '🎥 视频加载' : '📁 图像加载' }}
                </button>
            </div>

            <!-- (3) 标签列表 -->
            <div class="glass-card scrollable-card" style="max-height: 250px;">
                <div class="card-header">
                    <h4>🏷️ 标签列表</h4>
                    <button class="icon-btn-small" @click="isCreatingTag = true" title="新建标签">+</button>
                </div>
                
                <div v-if="isCreatingTag" class="new-entry-form mini-form">
                    <input type="text" v-model="newTagName" placeholder="标签名" @keyup.enter="confirmCreateTag">
                    <input type="color" v-model="newTagColor">
                    <div class="mini-actions">
                        <button class="small-btn primary" @click="confirmCreateTag">✓</button>
                        <button class="small-btn" @click="isCreatingTag = false">×</button>
                    </div>
                </div>

                <div class="list-container compact">
                    <div v-for="tag in tags" :key="tag.id" 
                         class="list-item" 
                         :class="{ active: selectedTagId === tag.id }"
                         @click="selectedTagId = tag.id">
                         
                        <div class="item-color" :style="{ background: tag.color }">
                            <input type="color" v-model="tag.color" class="color-picker-input">
                        </div>
                        <input type="text" v-model="tag.name" class="inline-edit-input" @click.stop>
                        
                        <div class="item-actions">
                            <button class="action-btn" @click.stop="toggleTagVisibility(tag)">{{ tag.visible ? '👁️' : '🙈' }}</button>
                            <button class="action-btn delete" @click.stop="deleteTag(tag.id)">🗑️</button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- (4) 标注与提示信息列表 (全模式共享) -->
            <div class="glass-card scrollable-card" style="flex:1; max-height: 300px;" v-if="['labeling', 'tracking', 'recognition'].includes(subTab)">
                <div class="card-header">
                    <h4>📝 目标列表</h4>
                </div>
                <div class="list-container compact">
                    <div v-for="ann in annotations" :key="ann.id" 
                         class="list-item ann-item"
                         @mouseenter="hoveredAnnId = ann.id"
                         @mouseleave="hoveredAnnId = null">
                        <div class="item-color" :style="{ background: getTagColor(ann.tagId) }"></div>
                        <div class="item-info">
                            <div class="ann-name" style="font-size: 13px;">{{ getTagName(ann.tagId) }}</div>
                        </div>
                        <button class="action-btn delete" @click.stop="deleteAnnotation(ann.id)">🗑️</button>
                    </div>
                    <div v-if="annotations.length === 0" class="empty-hint mini">尚未添加特征点/框</div>
                </div>

                <!-- 底部操作提示 (贴合列表底部) -->
                <div style="margin-top: auto; padding: 8px 10px; background: rgba(0,0,0,0.03); border-top: 1px solid rgba(0,0,0,0.05); font-size: 10px; color: #777; line-height: 1.4; border-bottom-left-radius: 12px; border-bottom-right-radius: 12px;">
                    左键正类锚点；右键负类锚点；拖拽画框；按住空格平移
                </div>
            </div>

            <!-- 识别结果展示及文字提示 -->
            <div class="glass-card recognition-result-card" v-if="subTab === 'recognition'">
                <div class="card-header"><h4>🔍 零样本文字/视觉检索</h4></div>
                <div class="recog-content" style="display: flex; flex-direction: column; gap: 8px; margin-top: 10px;">
                    <!-- 追加文本输入检索 -->
                    <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 2px;">
                        输入英文对象 (如 shoe) 进行多目标提取：
                    </div>
                    <div style="display: flex; gap: 8px; width: 100%;">
                        <input type="text" v-model="textPrompt" @keyup.enter="handleTextPromptSubmit" 
                               placeholder="对象名称..." 
                               style="flex: 1; min-width: 0; padding: 8px 12px; border-radius: 6px; border: 1px solid var(--panel-border); font-size: 13px; outline: none; background: rgba(255,255,255,0.7);"/>
                        <button class="floating-action-btn primary" style="padding: 0 12px; border-radius: 6px; font-size: 13px;" @click="handleTextPromptSubmit">识别</button>
                        <button class="floating-action-btn" style="padding: 0 12px; border-radius: 6px; font-size: 13px; background: #10B981; color: white;" @click="saveCurrentAnnotation" :disabled="!lastGeneratedMask">确认存入所有目标</button>
                    </div>
                    
                    <div style="display: flex; align-items: center; justify-content: space-between; font-size: 11px; color: var(--text-muted); margin-top: 2px;">
                        <span>多目标文本检出灵敏度: {{ textThreshold }}</span>
                        <input type="range" v-model="textThreshold" min="0.1" max="0.9" step="0.05" @change="handleTextPromptSubmit" style="width: 80px; accent-color: var(--primary-color);">
                    </div>
                    
                    <!-- 分割同类按钮：完全只依赖“目标列表”中已封入的掩码先验 -->
                    <div style="border-top: 1px dashed var(--panel-border); padding-top: 10px; margin-top: 5px;">
                        <div style="display: flex; align-items: center; justify-content: space-between; font-size: 12px; color: var(--text-muted); margin-bottom: 8px;">
                            <span>灵敏度阈值 (筛选同类用): {{ similarityThreshold }}</span>
                            <input type="range" v-model="similarityThreshold" min="0.1" max="0.9" step="0.05" style="width: 100px; accent-color: var(--primary-color);">
                        </div>
                        <button class="floating-action-btn secondary" @click="requestSimilarSeg" style="width:100%;" :disabled="annotations.length === 0">
                            ✨ 一键分割同类目标 (需列表有目标)
                        </button>
                    </div>
                    
                    <div style="font-size: 11px; color: var(--text-muted); text-align: center; margin-top: 5px;">
                        上方的提取将仅仅依托上方 "目标列表" 中的记录
                    </div>
                </div>
            </div>
            <!-- (视频追踪专用) 识别结果展示 -->
            <div class="glass-card recognition-result-card" v-if="subTab === 'tracking' && sessionId">
                <div class="card-header"><h4>🎞️ 视频追踪控制</h4></div>
                <div class="recog-content" style="display: flex; flex-direction: column; gap: 8px; margin-top: 10px;">
                    <button class="floating-action-btn primary" @click="startVideoTracking" style="width:100%" :disabled="isLoading || (annotations.length === 0 && !lastGeneratedMask)">
                        🏃 开始全量追踪
                    </button>
                    <!-- 新增：显式的产物查看按钮 -->
                    <button v-if="videoUrl" class="floating-action-btn" style="width:100%; background: #10B981; color: white; margin-top: 5px;" @click="showTrackingResult = true">
                        🎉 查看跟踪结果视频
                    </button>
                    <div style="font-size: 11px; color: var(--text-muted); text-align: center;">
                        在首帧画好框/点后，点击上方按钮启动视频追踪流
                    </div>
                </div>
            </div>
            
            <!-- (新增) 视频追踪结果列表 -->
            <div class="glass-card recognition-result-card" v-if="subTab === 'tracking'" style="flex:1; max-height: 250px; overflow-y: auto; margin-top: 10px;">
                <div class="card-header"><h4>📜 后台跟踪任务列</h4></div>
                <div class="list-container compact" style="margin-top: 10px;">
                    <div v-for="task in trackingTasks" :key="task.session_id" class="list-item" style="flex-direction: column; align-items: stretch; gap: 5px; padding: 10px; background: rgba(255,255,255,0.6); border: 1px solid rgba(0,0,0,0.05);">
                        <div style="display: flex; justify-content: space-between; width: 100%; align-items: center;">
                            <span style="font-size: 13px; font-weight: bold; color: var(--primary-color);">队列: {{ task.session_id.substring(0, 5) }}...</span>
                            <span style="font-size: 11px; padding: 3px 8px; border-radius: 4px;"
                                  :style="{background: task.status==='processing'?'#FEF3C7':task.status==='done'?'#D1FAE5':'#FEE2E2', 
                                           color: task.status==='processing'?'#D97706':task.status==='done'?'#059669':'#DC2626'}">
                                {{ task.status === 'processing' ? '运算中' : task.status === 'done' ? '已完成' : task.status === 'stopped' ? '已终止' : '错误' }}
                            </span>
                        </div>
                        <div v-if="task.status === 'processing'" style="width: 100%; background: #e0e0e0; height: 6px; border-radius: 3px; overflow: hidden; margin-top: 4px;">
                            <div :style="{width: ((task.progress / (task.totalFrames || 1)) * 100) + '%', background: '#D97706', height: '100%', transition: 'width 0.5s'}"></div>
                        </div>
                        <div v-if="task.status === 'processing'" style="font-size: 11px; color: var(--text-muted); text-align: right;">
                            进度: {{ ((task.progress / (task.totalFrames || 1)) * 100).toFixed(1) }}% ({{ task.progress }}/{{ task.totalFrames }})
                        </div>
                        
                        <div style="display: flex; gap: 8px; margin-top: 8px;">
                            <button v-if="task.status === 'done'" class="floating-action-btn primary" style="flex: 1; padding: 6px; font-size: 12px;" @click="previewTrackingResult(task.video_url)">🎬 播放</button>
                            <button v-if="task.status === 'processing'" class="floating-action-btn secondary" style="flex: 1; padding: 6px; color: #DC2626; font-size: 12px; border-color: #FCA5A5;" @click="stopOrDeleteTask(task.session_id)">🛑 中止</button>
                            <button v-else class="floating-action-btn secondary" style="flex: 1; padding: 6px; color: #DC2626; font-size: 12px; border-color: #FCA5A5;" @click="stopOrDeleteTask(task.session_id)">🗑️ 清除</button>
                        </div>
                    </div>
                    <div v-if="trackingTasks.length === 0" class="empty-hint mini">暂无历史跟踪记录</div>
                </div>
            </div>
            
        </div>

        <!-- 新增追踪结果大屏弹窗 (Modal) -->
        <div v-if="showTrackingResult" class="modal-overlay glass-overlay" style="z-index: 10000;" @click.self="showTrackingResult = false">
            <div class="modal-content glass-modal" style="max-width: 80%; width: auto;">
                <div class="modal-header">
                    <h3>🏆 视频追踪结果推流</h3>
                    <button class="close-btn" @click="showTrackingResult = false">×</button>
                </div>
                <div class="modal-body" style="padding: 10px;">
                    <video v-if="videoUrl" :src="videoUrl" controls autoplay loop style="max-width: 100%; max-height: 70vh;border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"></video>
                </div>
            </div>
        </div>

        <!-- 底部操作空间 (Z-Index: 1000) -->
        <div class="bottom-action-bar" style="z-index: 1000;">
            <!-- (5) 操作按钮 -->
            <div class="action-group" v-if="imageUrl || videoUrl">
                <div class="btn-group">
                    <button class="floating-action-btn primary" @click="saveCurrentAnnotation" v-if="['labeling', 'tracking', 'recognition'].includes(subTab)" :disabled="!lastGeneratedMask">
                        {{ subTab === 'tracking' ? '☑️ 确认并进入下一目标' : '💾 记录当前特征' }}
                    </button>
                    <button class="floating-action-btn secondary" @click="resetCurrentSession" :disabled="!sessionId">
                        🔄 清空标注
                    </button>
                </div>
            </div>
        </div>

        <!-- 图像/视频加载弹窗 (Modal) -->
        <div v-if="showUploadModal" class="modal-overlay glass-overlay" style="z-index: 9999;" @click.self="showUploadModal = false">
            <div class="modal-content glass-modal">
                <div class="modal-header">
                    <h3>🚀 {{ subTab === 'tracking' ? '初始化视频源' : '初始化图像源' }}</h3>
                    <button class="close-btn" @click="showUploadModal = false">×</button>
                </div>
                <div class="modal-body">
                    <div class="upload-area">
                        <label class="upload-btn">
                            📤 选择{{ subTab === 'tracking' ? '视频' : '文件' }}
                            <input type="file" :accept="subTab === 'tracking' ? 'video/*' : 'image/*'" @change="handleUpload" hidden>
                        </label>
                        <p class="upload-hint">支持拖拽文件到窗口</p>
                        <p class="upload-hint" style="color:#d946ef; font-weight:bold" v-if="subTab === 'tracking'">⚠️ 当前由于后端显存限制可能导致推理失败</p>
                    </div>
                    <ImageSource v-if="subTab !== 'tracking'" @image-selected="onImageSelected" @stream-frame="onStreamFrame" />
                </div>
            </div>
        </div>
    </div>
    `,

    setup() {
        const API_BASE = window.location.origin + "/api/sam";

        // --- 核心引用 ---
        const imageRef = ref(null);
        const videoRef = ref(null); // Added videoRef
        const canvasRef = ref(null); // Consolidated canvas ref

        // --- UI 状态 ---
        const showUploadModal = ref(false);
        const subTab = ref('labeling');
        const imageUrl = ref('');
        const videoUrl = ref(''); // Added videoUrl
        const sessionId = ref('');
        const statusMessage = ref('等待图像加载...');
        const isLoading = ref(false);

        // --- 标签与标注数据 ---
        const tags = ref([
            { id: '1', name: '默认目标', color: '#A21CAF', visible: true }
        ]);
        const selectedTagId = ref('1');
        const showTrackingResult = ref(false); // 控制最终追踪全景播放弹出层
        const trackingTasks = ref([]);
        let fetchTasksInterval = null;
        const isCreatingTag = ref(false);
        const newTagName = ref('');
        const newTagColor = ref('#A21CAF');
        const similarityThreshold = ref(0.2); // 一键同类的宽松拦截网
        const textThreshold = ref(0.4); // 文本找目标的专用阈值

        const annotations = ref([]); // { id, tagId, maskB64 }
        const hoveredAnnId = ref(null); // 上下文交互：当前鼠标所悬浮查看的标注 ID

        watch(subTab, (newTab, oldTab) => {
            if (newTab !== oldTab) {
                // 彻底清空所有数据
                imageUrl.value = '';
                videoUrl.value = '';
                sessionId.value = '';
                statusMessage.value = '等待图片/视频加载...';
                isLoading.value = false;
                annotations.value = [];
                textPrompt.value = '';
                recognitionResult.value = null;
                resetCurrentSession();
            }
        });

        watch(hoveredAnnId, () => {
            // 当鼠标在标注列表中进出悬浮时，重新绘制整个画布的所有图层，使悬浮特效(发光/加深)能渲染出来
            redrawAllMasks();
        });

        // --- 交互数据 ---
        const points = ref([]);
        const currentBox = ref(null);
        const dragBox = ref(null);
        const lastGeneratedMask = ref(null);
        const lastMultiMasksB64 = ref([]); // 专用于解构后的多目标独立遮罩，使得确认按钮可以分而治之
        const recognitionResult = ref(null); // { label, score }
        const textPrompt = ref(""); // 新增 textPrompt
        const currentFrameIdx = ref(0); // 记录当前视频播放帧
        const isDragging = ref(false);
        const dragStart = ref(null);
        const globalSpacePushed = ref(false);
        const isHintExpanded = ref(false); // 提示条折叠状态 (默认只有小灯泡)

        // 监听空格键
        const handleKeyDown = (e) => {
            if (e.code === 'Space') {
                globalSpacePushed.value = true;
                e.preventDefault(); // 防止页面滚动
            }
        };

        const handleKeyUp = (e) => {
            if (e.code === 'Space') {
                globalSpacePushed.value = false;
            }
        };

        onMounted(() => {
            if (!subTab.value) {
                subTab.value = 'labeling';
            }
            window.addEventListener('keydown', handleKeyDown);
            window.addEventListener('keyup', handleKeyUp);
            fetchTasks();
            fetchTasksInterval = setInterval(fetchTasks, 2000);
        });

        onUnmounted(() => {
            if (fetchTasksInterval) clearInterval(fetchTasksInterval);
            window.removeEventListener('keydown', handleKeyDown);
            window.removeEventListener('keyup', handleKeyUp);
        });



        // --- 标签管理逻辑 ---
        const confirmCreateTag = () => {
            if (!newTagName.value) return alert("请输入标签名");
            const id = Date.now().toString();
            tags.value.push({
                id,
                name: newTagName.value,
                color: newTagColor.value,
                visible: true
            });
            selectedTagId.value = id;
            isCreatingTag.value = false;
            newTagName.value = '';
        };

        const deleteTag = (id) => {
            if (tags.value.length <= 1) return alert("至少保留一个标签");
            tags.value = tags.value.filter(t => t.id !== id);
            if (selectedTagId.value === id) selectedTagId.value = tags.value[0].id;
        };

        const toggleTagVisibility = (tag) => {
            tag.visible = !tag.visible;
            redrawAllMasks();
        };

        const getTagColor = (tagId) => tags.value.find(t => t.id === tagId)?.color || '#999';
        const getTagName = (tagId) => tags.value.find(t => t.id === tagId)?.name || '未知';

        // --- 标注管理逻辑 ---
        const saveCurrentAnnotation = () => {
            if (!lastGeneratedMask.value || !selectedTagId.value) return;

            // 是否存在拆分后的多目标？（通常由带有效文本的预测，或者是同类分割返回）
            if (lastMultiMasksB64.value && lastMultiMasksB64.value.length > 0) {
                lastMultiMasksB64.value.forEach((b64, idx) => {
                    const objId = annotations.value.length + 1;
                    annotations.value.push({
                        id: Date.now().toString() + "_" + idx,
                        tagId: selectedTagId.value,
                        maskB64: b64,
                        objId: objId,
                        // 标记历史依据
                        savedPoints: JSON.parse(JSON.stringify(points.value)),
                        savedBox: currentBox.value ? JSON.parse(JSON.stringify(currentBox.value)) : null,
                        savedText: textPrompt.value
                    });
                });
                statusMessage.value = `共 ${lastMultiMasksB64.value.length} 个解离独立特征已被存入目标列表。`;
            } else {
                // 传统单件保存
                const objId = annotations.value.length + 1;
                annotations.value.push({
                    id: Date.now().toString(),
                    tagId: selectedTagId.value,
                    maskB64: lastGeneratedMask.value,
                    objId: objId,
                    savedPoints: JSON.parse(JSON.stringify(points.value)),
                    savedBox: currentBox.value ? JSON.parse(JSON.stringify(currentBox.value)) : null,
                    savedText: textPrompt.value
                });
                statusMessage.value = '特征已定型存入目标列表。可以开始绘制下一个目标或点击全量追踪。';
            }

            resetCurrentSession();
            redrawAllMasks();
        };

        const deleteAnnotation = (id) => {
            annotations.value = annotations.value.filter(a => a.id !== id);
            redrawAllMasks();
        };

        // --- 图像处理逻辑 ---
        const onImageSelected = (payload) => {
            const file = payload.data;
            if (!file) return;
            imageUrl.value = URL.createObjectURL(file);
            showUploadModal.value = false;
            uploadAndInitSession(file);
        };

        const onStreamFrame = async (b64) => {
            // 在标注模式下，如果尚未加载图片，允许通过流媒体首帧初始化
            if (!imageUrl.value && b64) {
                imageUrl.value = b64;
                const res = await fetch(b64);
                const blob = await res.blob();
                uploadAndInitSession(blob);
            }
            // 如果处于追踪模式且已就绪，实时处理流图像
            if (subTab.value === 'tracking' && sessionId.value && !isLoading.value) {
                requestPrediction();
            }
        };

        const handleUpload = async (event) => {
            const file = event.target.files[0];
            if (!file) return;

            if (subTab.value === 'tracking' && file.type.startsWith('video/')) {
                isLoading.value = true;
                statusMessage.value = "正在上传分析视频...";
                showUploadModal.value = false;

                // 本地预览
                videoUrl.value = URL.createObjectURL(file);
                imageUrl.value = null;
                resetCurrentSession();

                try {
                    const formData = new FormData();
                    formData.append('file', file);
                    // 1. 上传视频
                    const uploadRes = await axios.post(`${API_BASE}/video/upload`, formData, {
                        headers: { 'Content-Type': 'multipart/form-data' }
                    });

                    const videoPath = uploadRes.data.video_path;

                    // 2. 初始化 Tracker
                    statusMessage.value = "正在提取视频特征层...";
                    const startRes = await axios.post(`${API_BASE}/video/start_session`, {
                        video_path: videoPath
                    });

                    sessionId.value = startRes.data.session_id;
                    statusMessage.value = "视频及跟踪实例初始化完毕";
                } catch (e) {
                    console.error(e);
                    statusMessage.value = "视频加载失败，可能显存不足";
                } finally {
                    isLoading.value = false;
                }
                return;
            }

            // 处理普通图像
            const reader = new FileReader();
            reader.onload = async (e) => {
                const base64Data = e.target.result;
                videoUrl.value = null; // 抹除视频模式
                imageUrl.value = base64Data;
                showUploadModal.value = false;
                resetCurrentSession();

                try {
                    isLoading.value = true;
                    statusMessage.value = "正在解析图像拓扑及视觉语言特征...";
                    const base64_image = base64Data.split(',')[1];
                    const session_id = 'session_' + Math.random().toString(36).substr(2, 9);

                    const res = await axios.post(`${API_BASE}/upload`, {
                        image_base64: base64_image,
                        session_id: session_id
                    });

                    sessionId.value = res.data.session_id;
                    statusMessage.value = "模型处理完毕: 可以开始互助标注";
                } catch (e) {
                    console.error(e);
                    statusMessage.value = "模型内部错误";
                } finally {
                    isLoading.value = false;
                }
            };
            reader.readAsDataURL(file);
        };

        const uploadAndInitSession = async (file) => {
            isLoading.value = true;
            statusMessage.value = '正在提取图像特征...';
            const formData = new FormData();
            formData.append('file', file);
            try {
                const res = await axios.post(`${API_BASE}/upload`, formData);
                sessionId.value = res.data.session_id;
                statusMessage.value = '特征就绪，可以标注。';
            } catch (e) {
                statusMessage.value = '上传失败';
                console.error(e);
            } finally {
                isLoading.value = false;
            }
        };

        const onImageLoaded = () => {
            nextTick(() => {
                initCanvases();
            });
        };

        const handleVideoLoaded = () => {
            nextTick(() => {
                initCanvases();
            });
        };

        const initCanvases = () => {
            let mediaEl = null;

            if (subTab.value === 'tracking' && videoRef.value) {
                mediaEl = videoRef.value;
            } else if (imageRef.value) {
                mediaEl = imageRef.value;
            }

            if (!mediaEl || !canvasRef.value) return;

            // 获取媒体原始内容尺寸
            const nw = mediaEl.videoWidth || mediaEl.naturalWidth;
            const nh = mediaEl.videoHeight || mediaEl.naturalHeight;

            canvasRef.value.width = nw;
            canvasRef.value.height = nh;

            // 计算屏幕可用尺寸以撑满长边并居中
            if (canvasRef.value && canvasRef.value.parentElement) {
                // 父级 DOM 即 canvas-container
                const container = canvasRef.value.parentElement;
                container.style.width = `${nw}px`;
                container.style.height = `${nh}px`;

                const workspace = container.parentElement;
                if (workspace) {
                    const wsRect = workspace.getBoundingClientRect();
                    // 减去一定边距 padding
                    const maxW = wsRect.width - 120; // 左右留足够间隙供侧边栏
                    const maxH = wsRect.height - 120; // 上下留间隙供顶部漂浮条

                    const scaleX = maxW / nw;
                    const scaleY = maxH / nh;
                    // 以确保完全展示的最小缩放为基准
                    const optimalZoom = Math.min(scaleX, scaleY);

                    zoomLevel.value = optimalZoom;

                    // 基于中心放大策略和容器边界框测算偏移
                    panOffset.value = {
                        x: (wsRect.width - nw) / 2, // 注意：在 DOM 中 transform-origin 是 center，平移量只针对未经 scale 的框体进行置中即可
                        y: (wsRect.height - nh) / 2
                    };
                }
            }
        };

        // --- 缩放平移交互变量 ---
        const panOffset = ref({ x: 0, y: 0 });
        const zoomLevel = ref(1);
        const isPanDragging = ref(false);
        let panStartX = 0;
        let panStartY = 0;

        const handleWheel = (e) => {
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            const newScale = Math.max(0.1, Math.min(10, zoomLevel.value * delta));
            // Center scaling approx
            zoomLevel.value = newScale;
        };

        const startPan = (e) => {
            if (e.button === 1 || globalSpacePushed.value) {
                isPanDragging.value = true;
                panStartX = e.clientX - panOffset.value.x;
                panStartY = e.clientY - panOffset.value.y;
            }
        };

        const doPan = (e) => {
            if (!isPanDragging.value) return;
            panOffset.value.x = e.clientX - panStartX;
            panOffset.value.y = e.clientY - panStartY;
        };

        const endPan = () => {
            isPanDragging.value = false;
        };

        const resetViewport = () => {
            zoomLevel.value = 1.0;
            panOffset.value = { x: 0, y: 0 };
        };

        // --- 交互预测逻辑 ---
        const getCanvasMousePos = (e) => {
            const canvas = canvasRef.value;
            const rect = canvas.getBoundingClientRect();
            // 在缩放平移后的 Canvas 包围盒坐标系中定位
            const x = (e.clientX - rect.left) * (canvas.width / rect.width);
            const y = (e.clientY - rect.top) * (canvas.height / rect.height);
            return { x, y };
        };

        const handleMouseDown = (e) => {
            // 如果是在进行视口拖拽（中键或空格/Alt+左键），不触发标注模式
            if (e.button === 1 || (e.button === 0 && (e.altKey || globalSpacePushed.value))) {
                return;
            }

            if (!sessionId.value || isLoading.value) return;
            const pos = getCanvasMousePos(e);

            if (e.button === 2) { // 右键负提示
                points.value.push({ x: pos.x, y: pos.y, label: 0 });
                redrawPrompts();
                requestPrediction();
                return;
            }

            isDragging.value = true;
            dragStart.value = pos;
        };

        const handleMouseMove = (e) => {
            if (!isDragging.value) return;
            const pos = getCanvasMousePos(e);
            dragBox.value = {
                xmin: Math.min(dragStart.value.x, pos.x),
                ymin: Math.min(dragStart.value.y, pos.y),
                xmax: Math.max(dragStart.value.x, pos.x),
                ymax: Math.max(dragStart.value.y, pos.y)
            };
            redrawPrompts();
        };

        const handleMouseUp = (e) => {
            if (!isDragging.value) return;
            isDragging.value = false;
            const pos = getCanvasMousePos(e);
            const dist = Math.sqrt(Math.pow(pos.x - dragStart.value.x, 2) + Math.pow(pos.y - dragStart.value.y, 2));

            if (dist < 10) { // 点击
                points.value.push({ x: pos.x, y: pos.y, label: 1 });
            } else { // 框选
                currentBox.value = {
                    xmin: Math.min(dragStart.value.x, pos.x),
                    ymin: Math.min(dragStart.value.y, pos.y),
                    xmax: Math.max(dragStart.value.x, pos.x),
                    ymax: Math.max(dragStart.value.y, pos.y)
                };
                // 一旦框选（即确立了目标所在的主体边界），清空之前对该对象的点选记录，以新的边界框作为起底
                points.value = [];
            }
            dragBox.value = null;
            redrawPrompts();
            requestPrediction();
        };

        const requestPrediction = async () => {
            try {
                if (!sessionId.value) return;
                isLoading.value = true;
                const targetColor = getTagColor(selectedTagId.value);

                // HEX -> [B, G, R, A]，对于 OpenCV 处理遮罩来说通常是 BGR
                const hexToBgr = (hex) => {
                    const cleanHex = (hex || '#2ecc71').replace('#', '');
                    const r = parseInt(cleanHex.slice(0, 2), 16);
                    const g = parseInt(cleanHex.slice(2, 4), 16);
                    const b = parseInt(cleanHex.slice(4, 6), 16);
                    return [b, g, r, 150]; // 默认加点透明度作为掩码底色返回
                };

                // 如果处于识别模式且有点，则同时请求识别
                if (subTab.value === 'recognition' && points.value && points.value.length > 0) {
                    requestRecognition();
                }

                try {
                    let maskBase64 = null;
                    if (subTab.value === 'tracking') {
                        // Video Tracking Request
                        const currentObjId = annotations.value.length + 1;
                        const res = await axios.post(`${API_BASE}/video/add_prompt`, {
                            session_id: sessionId.value,
                            frame_idx: currentFrameIdx.value,
                            obj_id: currentObjId,
                            points: points.value,
                            boxes: currentBox.value ? [currentBox.value] : [],
                            text: textPrompt.value || "",
                            mask_color: hexToBgr(targetColor),
                            v_width: canvasRef.value.width,
                            v_height: canvasRef.value.height
                        });

                        maskBase64 = res.data.mask_base64;
                        if (maskBase64) {
                            lastGeneratedMask.value = maskBase64;
                            drawCurrentMask();
                        }
                        statusMessage.value = `视频流：目标 ${currentObjId} 特征已记录，请确认定型。`;
                    } else {
                        // Static Image Recognition/Segmenting Request
                        const targetColor = getTagColor(selectedTagId.value);
                        const res = await axios.post(`${API_BASE}/predict`, {
                            session_id: sessionId.value,
                            points: points.value,
                            boxes: currentBox.value ? [currentBox.value] : [],
                            text: textPrompt.value || "",
                            mask_color: hexToBgr(targetColor),
                            v_width: canvasRef.value.width,
                            v_height: canvasRef.value.height,
                            text_threshold: parseFloat(textThreshold.value)
                        });
                        maskBase64 = res.data.mask_base64;
                        lastMultiMasksB64.value = res.data.multi_masks_base64 || [];
                        statusMessage.value = `掩码已高亮展示 (识别到 ${lastMultiMasksB64.value.length || 1} 个子片)。`;
                        lastGeneratedMask.value = maskBase64;
                        drawCurrentMask();
                    }
                } catch (e) {
                    console.error("生成失败", e);
                    statusMessage.value = '生成追踪失败，可能网络中断。';
                } finally {
                    isLoading.value = false;
                }
            } catch (outerE) {
                console.error("预测请求外部执行栈报错:", outerE);
                isLoading.value = false;
            }
        };

        const fetchTasks = async () => {
            // 实装前端空闲轮询避让：如果不在 tracking 面板且没有任何进行中的任务，停止拉取以免污染后端日志
            const hasActive = trackingTasks.value.some(t => t.status === 'processing');
            if (!hasActive && subTab.value !== 'tracking') {
                return;
            }
            try {
                const res = await axios.get(`${API_BASE}/video/tasks`);
                trackingTasks.value = res.data;
                const active = res.data.find(t => t.session_id === sessionId.value && t.status === 'processing');
                if (active) {
                    statusMessage.value = `正在后台流式计算掩码... 已处理到第 ${active.progress} 帧 / 共 ${active.totalFrames || '?'} 帧`;
                } else if (isLoading.value && sessionId.value) {
                    isLoading.value = false;
                    statusMessage.value = "追踪任务更新完毕。";
                }
            } catch (e) {
                console.error("Fetch tasks error", e);
            }
        };

        const stopOrDeleteTask = async (sid) => {
            try {
                await axios.delete(`${API_BASE}/video/tasks/${sid}`);
                fetchTasks();
            } catch (e) {
                console.error("Delete task failed", e);
            }
        };

        const previewTrackingResult = (url) => {
            videoUrl.value = url + "?t=" + Date.now();
            showTrackingResult.value = true;
        };

        const startVideoTracking = async () => {
            if (!sessionId.value) return;
            isLoading.value = true;
            statusMessage.value = "正在移交服务器启动异步全局追踪...";
            try {
                // 这个接口将会秒回，因为核心逻辑已下发到 FastAPI 的多线程资源池
                await axios.post(`${API_BASE}/video/propagate`, {
                    session_id: sessionId.value
                });
                fetchTasks();
            } catch (e) {
                console.error("追踪启动失败", e);
                statusMessage.value = "追踪启动请求失败";
                isLoading.value = false;
            }
        };

        const handleTextPromptSubmit = () => {
            if (!textPrompt.value.trim() || !sessionId.value) return;
            statusMessage.value = "正在基于文本搜寻目标(支持多目标过滤)...";
            requestPrediction();
            // 在缺乏 VLM 的此版本，不抛出假定的文字推论置信度显示
        };

        const hexToBgr = (hex) => {
            const color = hex.replace('#', '');
            const r = parseInt(color.substring(0, 2), 16);
            const g = parseInt(color.substring(2, 4), 16);
            const b = parseInt(color.substring(4, 6), 16);
            return [b, g, r, 153];
        };

        const requestSimilarSeg = async () => {
            if (!sessionId.value) return;
            if (annotations.value.length === 0) {
                statusMessage.value = "目标列表为空，无法进行同类目标扩展寻找。";
                return;
            }
            statusMessage.value = "正在提取目标列表特征，开启全图同类搜寻...";

            // 提取时彻头彻尾剥离当前画板输入，100% 只用列表内的已知目标来作为范本
            let allPoints = [];
            let allBox = [];
            let allTexts = [];

            annotations.value.forEach(ann => {
                if (ann.savedPoints) allPoints.push(...ann.savedPoints);
                if (ann.savedBox && ann.savedBox.xmin !== undefined) allBox.push(ann.savedBox);
                if (ann.savedText && ann.savedText.trim() !== '' && !allTexts.includes(ann.savedText)) allTexts.push(ann.savedText);
            });

            try {
                isLoading.value = true;
                const targetColor = getTagColor(selectedTagId.value);
                const res = await axios.post(`${API_BASE}/predict`, {
                    session_id: sessionId.value,
                    points: allPoints,
                    boxes: allBox,
                    text: allTexts.join(", "),
                    mask_color: hexToBgr(targetColor),
                    v_width: canvasRef.value.width,
                    v_height: canvasRef.value.height,
                    find_similar: true,
                    similarity_threshold: parseFloat(similarityThreshold.value),
                    text_threshold: parseFloat(textThreshold.value)
                });

                if (res.data.mask_base64) {
                    lastGeneratedMask.value = res.data.mask_base64;
                    lastMultiMasksB64.value = res.data.multi_masks_base64 || [];
                    drawCurrentMask();
                }
                statusMessage.value = `同类关联提取已完成 (共聚合 ${lastMultiMasksB64.value.length || 1} 个碎片)。满意请点 [确认存入]。`;
            } catch (e) {
                console.error("生成同类失败", e);
                statusMessage.value = '寻找近亲族群掩码异常。';
            } finally {
                isLoading.value = false;
            }
        };

        const requestRecognition = async () => {
            // 原 identify 接口已弃用
        };

        const triggerRender = () => {
            if (!canvasRef.value) return;
            const ctx = canvasRef.value.getContext('2d');
            const imagesToLoad = [];

            // 1. Saved annotations
            annotations.value.forEach(ann => {
                const tag = tags.value.find(t => t.id === ann.tagId);
                if (tag && tag.visible && ann.maskB64) {
                    imagesToLoad.push({ src: ann.maskB64, type: 'saved', ann, tag });
                }
            });

            // 2. Current mask
            if (lastGeneratedMask.value) {
                imagesToLoad.push({ src: lastGeneratedMask.value, type: 'current' });
            }

            const drawSyncPrompts = () => {
                if (dragBox.value) {
                    ctx.strokeStyle = '#f1c40f';
                    ctx.setLineDash([10, 5]);
                    ctx.strokeRect(dragBox.value.xmin, dragBox.value.ymin, dragBox.value.xmax - dragBox.value.xmin, dragBox.value.ymax - dragBox.value.ymin);
                    ctx.setLineDash([]);
                }
                if (currentBox.value) {
                    ctx.strokeStyle = '#f1c40f';
                    ctx.lineWidth = 2;
                    ctx.strokeRect(currentBox.value.xmin, currentBox.value.ymin, currentBox.value.xmax - currentBox.value.xmin, currentBox.value.ymax - currentBox.value.ymin);
                }
                points.value.forEach(pt => {
                    ctx.beginPath();
                    ctx.arc(pt.x, pt.y, 5, 0, Math.PI * 2);
                    ctx.fillStyle = pt.label === 1 ? '#2ecc71' : '#e74c3c';
                    ctx.fill();
                    ctx.strokeStyle = '#fff';
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                });
            };

            const performDraw = () => {
                ctx.clearRect(0, 0, canvasRef.value.width, canvasRef.value.height);
                // Draw saved
                imagesToLoad.filter(i => i.type === 'saved').forEach(i => {
                    if (i.img) {
                        ctx.globalAlpha = (hoveredAnnId.value === i.ann.id) ? 1.0 : 0.7;
                        if (hoveredAnnId.value === i.ann.id) {
                            ctx.shadowColor = i.tag.color;
                            ctx.shadowBlur = 15;
                        }
                        ctx.drawImage(i.img, 0, 0);
                        ctx.shadowBlur = 0;
                        ctx.globalAlpha = 1.0;
                    }
                });
                // Draw current
                const current = imagesToLoad.find(i => i.type === 'current');
                if (current && current.img) {
                    ctx.globalAlpha = 0.6;
                    ctx.drawImage(current.img, 0, 0);
                    ctx.globalAlpha = 1.0;
                }
                drawSyncPrompts();
            };

            if (imagesToLoad.length === 0) {
                ctx.clearRect(0, 0, canvasRef.value.width, canvasRef.value.height);
                drawSyncPrompts();
                return;
            }

            let loadedCount = 0;
            imagesToLoad.forEach(item => {
                const img = new Image();
                img.onload = () => {
                    item.img = img;
                    loadedCount++;
                    if (loadedCount === imagesToLoad.length) performDraw();
                };
                img.onerror = () => {
                    console.warn("图像加载失败:", item.src);
                    loadedCount++;
                    if (loadedCount === imagesToLoad.length) performDraw();
                };
                img.src = item.src;
            });
        };

        const drawCurrentMask = triggerRender;
        const redrawAllMasks = triggerRender;
        const redrawPrompts = triggerRender;

        const resetCurrentSession = () => {
            points.value = [];
            currentBox.value = null;
            lastGeneratedMask.value = '';
            lastMultiMasksB64.value = []; // Reset multi-masks as well
            recognitionResult.value = null;

            if (!canvasRef.value) return;
            const ctx = canvasRef.value.getContext('2d');
            ctx.clearRect(0, 0, canvasRef.value.width, canvasRef.value.height);
            redrawPrompts();
        };

        onMounted(() => {
            window.addEventListener('keydown', (e) => {
                if (e.key.toLowerCase() === 's' && (e.ctrlKey || e.metaKey)) {
                    e.preventDefault();
                    saveCurrentAnnotation();
                }
                if (e.key.toLowerCase() === 'r') resetCurrentSession();
            });
        });

        // ================= 模块导出 =================
        return {
            subTab, showUploadModal, imageRef, videoRef, canvasRef,
            imageUrl, videoUrl, sessionId, statusMessage, isLoading,
            tags, selectedTagId, isCreatingTag, newTagName, newTagColor, annotations, hoveredAnnId,
            zoomLevel, panOffset, globalSpacePushed, isPanDragging, isHintExpanded,
            lastGeneratedMask, lastMultiMasksB64, recognitionResult, currentFrameIdx, textPrompt,
            handleWheel, startPan, doPan, endPan,
            onImageSelected, onStreamFrame, handleVideoLoaded, onImageLoaded, saveCurrentAnnotation, resetCurrentSession,
            handleMouseDown, handleMouseMove, handleMouseUp, requestPrediction, requestRecognition, requestSimilarSeg, startVideoTracking,
            confirmCreateTag, deleteTag, toggleTagVisibility, getTagColor, getTagName, deleteAnnotation,
            handleUpload, handleTextPromptSubmit, textThreshold,
            points, currentBox, dragBox, showTrackingResult, similarityThreshold,
            trackingTasks, stopOrDeleteTask, previewTrackingResult
        };
    }
};
