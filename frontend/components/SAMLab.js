import { ref, onMounted, onUnmounted, nextTick, computed, watch, watchEffect } from 'vue';
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
        <div class="left-floating-panel" style="z-index: 1000;" :class="{'collapsed': !isLeftPanelExpanded}">
            
            <button class="toggle-panel-btn left-toggle" @click="isLeftPanelExpanded = !isLeftPanelExpanded" :title="isLeftPanelExpanded ? '收起左侧面板' : '展开左侧面板'">
                {{ isLeftPanelExpanded ? '◀' : '▶' }}
            </button>
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
                    </div>
                    
                    <!-- 分割同类按钮：完全只依赖“目标列表”中已封入的掩码先验 -->
                    <div style="border-top: 1px dashed var(--panel-border); padding-top: 10px; margin-top: 5px; display: flex; justify-content: center;">
                        <button class="floating-action-btn secondary" @click="requestSimilarSeg" style="width: 80%;" :disabled="annotations.length === 0">
                            ✨ 一键分割
                        </button>
                    </div>

                    <div style="display: flex; align-items: center; justify-content: space-between; font-size: 11px; color: var(--text-muted); margin-top: 10px;">
                        <span style="flex-shrink: 0;">置信度阈值: {{ confidenceThreshold }}</span>
                        <input type="range" v-model="confidenceThreshold" max="1" min="0" step="0.2" @change="handleThresholdChange" style="flex: 1; margin: 0 8px; accent-color: var(--primary-color);">
                    </div>
                    
                    <!-- 全部的全局过滤参数 -->
                    <div style="border-top: 1px solid rgba(0,0,0,0.05); padding-top: 10px; margin-top: 5px;">
                        <div style="display: flex; align-items: center; justify-content: space-between; font-size: 11px; color: #EAB308;">
                            <span>⚠️ 排他 IOU 过滤阈值: {{ iouThreshold }}</span>
                            <input type="range" v-model="iouThreshold" min="0" max="1" step="0.2" @change="applyIouFilter" style="width: 100px; accent-color: #EAB308;">
                        </div>
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
            
        </div>
        
        <!-- 右侧数据全景大屏 (Z-Index: 1000) -->
        <div class="right-floating-panel" style="z-index: 1000;" :class="{'collapsed': !isRightPanelExpanded}">
        
            <button class="toggle-panel-btn right-toggle" @click="isRightPanelExpanded = !isRightPanelExpanded" :title="isRightPanelExpanded ? '收起右侧数据大屏' : '展开右侧数据大屏'">
                {{ isRightPanelExpanded ? '▶' : '◀' }}
            </button>
            <!-- 标注与提示信息列表 (全模式共享) -->
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
                    <div v-if="annotations.length === 0" class="empty-hint mini">尚未存入任何已确立目标</div>
                </div>
                 <!-- 底部操作提示 (贴合列表底部) -->
                <div style="margin-top: auto; padding: 8px 10px; background: rgba(0,0,0,0.03); border-top: 1px solid rgba(0,0,0,0.05); font-size: 10px; color: #777; line-height: 1.4; border-bottom-left-radius: 12px; border-bottom-right-radius: 12px;">
                    画布左键正类；右键负类；拖拽画框；长按空格平移
                </div>
            </div>

            <!-- 待确认掩码池 (Pending List) -->
            <div class="glass-card scrollable-card" style="flex:1; max-height: 250px;" v-if="lastMultiMasksB64 && lastMultiMasksB64.length > 0">
                <div class="card-header" style="justify-content: space-between; align-items: center; display: flex;">
                    <h4>⏳ 待确认列表 ({{ lastMultiMasksB64.length }})</h4>
                    <div style="display: flex; gap: 4px;">
                        <button class="small-btn primary" style="background:#10B981; border-color:#059669;" @click="confirmMultiTargets()" title="一键将下方所有项转为确立目标">一键确认</button>
                        <button class="small-btn" style="background:#ef4444; color:white; border:none;" @click="clearPending" title="立刻清空所有的残影">清空</button>
                    </div>
                </div>
                <div class="list-container compact">
                    <div v-for="(item, idx) in lastMultiMasksB64" :key="idx" 
                         class="list-item ann-item"
                         :class="{ 'hovered-item': hoveredPendingIdx === idx }"
                         @mouseover="hoveredPendingIdx = idx"
                         @mouseleave="hoveredPendingIdx = null"
                         style="flex-direction: column; align-items: stretch; gap: 4px; position: relative;"
                         :style="{ borderLeft: '3px solid ' + getTagColor(selectedTagId) }">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div class="ann-name" style="font-size: 12px; color: #333; font-weight: bold;">{{ getTagName(selectedTagId) }} (残影 #{{ idx + 1 }})</div>
                            <div style="display: flex; gap: 4px;">
                                <button class="action-btn" style="background:#10B981; color:white; padding: 2px 6px; border-radius: 4px; font-size: 11px;" @click.stop="confirmSingleTarget(idx)">确认</button>
                                <button class="action-btn" style="background:#ef4444; color:white; padding: 2px 6px; border-radius: 4px; font-size: 11px;" @click.stop="cancelSingleTarget(idx)">取消</button>
                            </div>
                        </div>
                        <div style="display: flex; gap: 10px; font-size: 10px; color: var(--text-muted); background: rgba(255,255,255,0.4); padding: 3px 6px; border-radius: 4px;">
                            <span>置信得分:<b style="color:#059669;">{{ (item.score * 100).toFixed(1) }}%</b></span>
                            <span v-if="item.iou !== undefined">碰撞 IOU:<b style="color:#DC2626;">{{ (item.iou * 100).toFixed(1) }}%</b></span>
                        </div>
                    </div>
                </div>
                <div style="margin-top: auto; padding: 6px 10px; background: rgba(234,179,8,0.1); border-top: 1px dashed rgba(234,179,8,0.3); font-size: 10px; color: #B45309; line-height: 1.4; border-bottom-left-radius: 12px; border-bottom-right-radius: 12px;">
                    调节左侧的置信度和IOU滑块可实时剔除劣质或重叠的残影。
                </div>
            </div>

            <!-- 视频追踪结果列表 -->
            <div class="glass-card recognition-result-card" v-if="subTab === 'tracking'" style="flex:1; max-height: 250px; overflow-y: auto;">
                <div class="card-header"><h4>📜 后台跟踪队列</h4></div>
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
                    <video v-if="trackingResultUrl" :src="trackingResultUrl" controls autoplay loop style="max-width: 100%; max-height: 70vh;border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"></video>
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
                    <div class="upload-area" v-if="subTab === 'tracking'">
                        <p class="upload-hint" style="color:#d946ef; font-weight:bold; margin-bottom: 10px;">⚠️ 当前由于硬件限制可能视频特征抽取与追踪过程会比较慢</p>
                    </div>
                    <ImageSource @image-selected="onImageSelected" @stream-frame="onStreamFrame" @video-selected="handleUploadFromComponent" @local-video-selected="handleLocalVideo" :defaultSourceType="subTab === 'tracking' ? 'local_video' : 'upload'" :disableStreaming="subTab === 'tracking'" />
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
        const isLeftPanelExpanded = ref(true);
        const isRightPanelExpanded = ref(true);
        const showUploadModal = ref(false);
        const subTab = ref('labeling');
        const imageUrl = ref('');
        const videoUrl = ref(''); // Added videoUrl
        const trackingResultUrl = ref(''); // 弹出层视频结果地址
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
        const confidenceThreshold = ref(0.4); // 统一的置信度大闸门
        const iouThreshold = ref(0.4); // 用于过滤多目标生成时过于重叠的碎片

        const annotations = ref([]); // { id, tagId, maskB64 }
        const hoveredAnnId = ref(null); // 上下文交互：当前鼠标所悬浮查看的标注 ID
        const hoveredPendingIdx = ref(null); // 上下文交互：目前正在鼠标悬浮看哪个残影项

        let flashAnimationId = null; // 控制闪烁动画帧的全局指针

        watch(subTab, (newTab, oldTab) => {
            if (newTab !== oldTab) {
                // 彻底清空所有数据
                imageUrl.value = '';
                videoUrl.value = '';
                sessionId.value = '';
                statusMessage.value = '等待图片/视频加载...';
                isLoading.value = false;
                annotations.value = [];
                maskDataCache.clear();
                textPrompt.value = '';
                recognitionResult.value = null;
                resetCurrentSession();

                if (flashAnimationId) {
                    cancelAnimationFrame(flashAnimationId);
                    flashAnimationId = null;
                }
            }
        });

        // 【新增 6】：切换图片/视频后清空标注，确保新任务不携带旧残余
        watch([imageUrl, videoUrl], () => {
            if (imageUrl.value || videoUrl.value) {
                console.log("Detecting source change, clearing annotations and cache...");
                annotations.value = [];
                maskDataCache.clear();
                lastGeneratedMask.value = '';
                rawMultiMasksB64.value = [];
                lastMultiMasksB64.value = [];
                textPrompt.value = '';
            }
        });

        watch(hoveredAnnId, () => {
            // 当鼠标在标注列表中进出悬浮时，重新绘制整个画布的所有图层，使悬浮特效(发光/加深)能渲染出来
            redrawAllMasks();
        });

        watch(hoveredPendingIdx, (newVal) => {
            // 清理旧帧
            if (flashAnimationId !== null) {
                cancelAnimationFrame(flashAnimationId);
                flashAnimationId = null;
            }
            // 触发一次重绘即可。如果 newVal !== null，内部的 performDraw(拿到缓存好图片的闭包) 会自动接管帧循环
            redrawAllMasks();
        });
        // --- 交互数据 ---
        const points = ref([]);
        const currentBox = ref(null);
        const dragBox = ref(null);
        const lastGeneratedMask = ref(null);
        const rawMultiMasksB64 = ref([]); // 来自后端未被 IOU 过滤的原始多目标遮罩合集
        const lastMultiMasksB64 = ref([]); // 经过 IOU 过滤后，专门用于渲染确认发车的遮罩合集
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
            fetchTasks(); // 初始化只拉一次以恢复之前的断点记录
        });

        onUnmounted(() => {
            stopPolling();
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

        const maskDataCache = new Map(); // 用于存储已保存标注的 ImageData 缓存

        const b64ToCanvas = (base64) => {
            if (!base64) return Promise.resolve(null);
            return new Promise((resolve) => {
                const img = new Image();
                img.onload = () => {
                    const can = document.createElement('canvas');
                    can.width = img.width;
                    can.height = img.height;
                    const ctx = can.getContext('2d', { willReadFrequently: true });
                    ctx.drawImage(img, 0, 0);
                    resolve(ctx.getImageData(0, 0, img.width, img.height).data);
                };
                img.onerror = () => {
                    console.error("Failed to load mask image for IOU calculation.");
                    resolve(null);
                };
                // 兼容性检查：如果 base64 已经包含了 data: 协议头，则不需要重复叠加
                if (base64.startsWith('data:')) {
                    img.src = base64;
                } else {
                    img.src = "data:image/png;base64," + base64;
                }
            });
        };

        const calcIOU = (dataA, dataB) => {
            if (!dataA || !dataB) return 0;
            if (dataA.length !== dataB.length) {
                console.warn("IOU Dimension Mismatch:", dataA.length, dataB.length);
                // 兜底：如果尺寸不匹配，说明跨图片了，IOU 理论上就是 0
                return 0;
            }
            let intersection = 0;
            let union = 0;
            // 每4个元素为一个像素 (r,g,b,a)，仅看alpha
            for (let i = 3; i < dataA.length; i += 4) {
                const aFilled = dataA[i] > 10; // 稍微抬高阈值规避杂色
                const bFilled = dataB[i] > 10;
                if (aFilled && bFilled) intersection++;
                if (aFilled || bFilled) union++;
            }
            const iou = union === 0 ? 0 : intersection / union;
            console.log("Calculated IOU:", iou, "Intersection:", intersection, "Union:", union);
            return iou;
        };

        const applyIouFilter = async () => {
            if (!rawMultiMasksB64.value || rawMultiMasksB64.value.length === 0) {
                lastMultiMasksB64.value = [];
                drawCurrentMask();
                return;
            }

            // 第一层清洗：置信分大闸 (直接过滤 score 低于 confidenceThreshold 的残片)
            const confidencePassed = rawMultiMasksB64.value.filter(
                cand => cand.score >= parseFloat(confidenceThreshold.value)
            );

            if (annotations.value.length === 0) {
                // 如果没有已存在的实体，无需计算 IOU，只需记录 score 即可展示
                lastMultiMasksB64.value = confidencePassed.map(c => ({ ...c, iou: 0 }));
            } else {
                statusMessage.value = '正在运算像素级重叠及置信联排...';
                const existDatas = [];
                for (const a of annotations.value) {
                    // 优先从缓存读取
                    if (maskDataCache.has(a.id)) {
                        existDatas.push(maskDataCache.get(a.id));
                        continue;
                    }
                    const data = await b64ToCanvas(a.maskB64);
                    if (data) {
                        maskDataCache.set(a.id, data);
                        existDatas.push(data);
                    }
                }

                const accepted = [];
                for (const candidate of confidencePassed) {
                    // candidate.mask_base64 此时不应进缓存，它是瞬时的
                    const candidateData = await b64ToCanvas(candidate.mask_base64);
                    if (!candidateData) {
                        // 如果候选图加载失败，我们还是保守地允许它预览，但 iou 计为 0
                        accepted.push({ ...candidate, iou: 0 });
                        continue;
                    }

                    let maxIOU = 0;
                    for (const existData of existDatas) {
                        const iou = calcIOU(candidateData, existData);
                        if (iou > maxIOU) maxIOU = iou;
                    }

                    if (maxIOU <= parseFloat(iouThreshold.value)) {
                        accepted.push({
                            ...candidate,
                            iou: maxIOU
                        });
                    }
                }
                lastMultiMasksB64.value = accepted;

                const dropCount = confidencePassed.length - accepted.length;
                if (dropCount > 0) {
                    statusMessage.value = `生成完毕。IOU 阈值排他过滤已自动拦截 ${dropCount} 个与已有目标高度重叠的重复残影。`;
                } else {
                    statusMessage.value = '特征已生成并进入待确认列表。';
                }
            }
            drawCurrentMask();
        };

        const handleThresholdChange = () => {
            applyIouFilter();
        };

        const confirmMultiTargets = () => {
            if (lastMultiMasksB64.value && lastMultiMasksB64.value.length > 0) {
                lastMultiMasksB64.value.forEach((item, idx) => {
                    const objId = annotations.value.length + 1;
                    annotations.value.push({
                        id: Date.now().toString() + "_" + idx,
                        tagId: selectedTagId.value,
                        maskB64: item.mask_base64,
                        objId: objId,
                        // 标记历史依据
                        savedPoints: JSON.parse(JSON.stringify(points.value)),
                        savedBox: currentBox.value ? JSON.parse(JSON.stringify(currentBox.value)) : null,
                        savedText: textPrompt.value
                    });
                });
                statusMessage.value = `共 ${lastMultiMasksB64.value.length} 个独立特征已被存入列表！`;
            }
            resetCurrentSession();
            redrawAllMasks();
        };

        const confirmSingleTarget = (idx) => {
            if (lastMultiMasksB64.value && lastMultiMasksB64.value[idx]) {
                const item = lastMultiMasksB64.value[idx];
                const objId = annotations.value.length + 1;
                annotations.value.push({
                    id: Date.now().toString() + "_s_" + idx,
                    tagId: selectedTagId.value,
                    maskB64: item.mask_base64,
                    objId: objId,
                    savedPoints: JSON.parse(JSON.stringify(points.value)),
                    savedBox: currentBox.value ? JSON.parse(JSON.stringify(currentBox.value)) : null,
                    savedText: textPrompt.value
                });
                // 从待确认池剔除该体
                lastMultiMasksB64.value.splice(idx, 1);

                statusMessage.value = `单目标已被确立。剩余 ${lastMultiMasksB64.value.length} 个候选。`;

                // 如果待确定池空了，自动收尾这把
                if (lastMultiMasksB64.value.length === 0) {
                    resetCurrentSession();
                } else {
                    drawCurrentMask();
                }
                redrawAllMasks();
            }
        };

        const cancelSingleTarget = (idx) => {
            if (lastMultiMasksB64.value && lastMultiMasksB64.value.length > idx) {
                lastMultiMasksB64.value.splice(idx, 1);
                statusMessage.value = `单目标已被抛弃。剩余 ${lastMultiMasksB64.value.length} 个候选。`;
                if (lastMultiMasksB64.value.length === 0) {
                    resetCurrentSession();
                } else {
                    drawCurrentMask();
                }
                redrawAllMasks();
            }
        };

        const clearPending = () => {
            lastMultiMasksB64.value = [];
            statusMessage.value = "已清空所有待确认候补列表。";
            resetCurrentSession();
            redrawAllMasks();
        };

        // --- 取消通用的单一 save 的多目标能力（专属抽离到上述） ---
        const saveCurrentAnnotation = () => {
            if (!lastGeneratedMask.value || !selectedTagId.value) return;
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

            resetCurrentSession();
            redrawAllMasks();
        };

        const deleteAnnotation = (id) => {
            annotations.value = annotations.value.filter(a => a.id !== id);
            maskDataCache.delete(id); // 同时清理缓存
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

        const onStreamFrame = async (b64) => { // ImageSource 组件传来的视频帧
            if (!b64 || subTab.value === 'tracking') return; // 如果处于追踪模式，坚决拒收流媒体帧（防闪屏）

            // 新逻辑：只把摄像头画面当成镜子 (Mirror preview) 给用户看，绝对不私自发起 AI 识别，
            // 直到用户在 ImageSource 组件点击了“拍照并应用”！
            imageUrl.value = b64.includes('data:') ? b64 : "data:image/jpeg;base64," + b64.split(',').pop();

            // 旧版的自动识别逻辑全部移除！
        };

        const handleUploadFromComponent = (payload) => {
            const file = payload.data;
            if (!file) return;
            handleUpload({ target: { files: [file] } });
        };

        const handleLocalVideo = async (payload) => {
            const fileName = payload.data || payload;
            if (!fileName || typeof fileName !== 'string') return;

            if (subTab.value === 'tracking') {
                isLoading.value = true;
                statusMessage.value = "正在拉取本地置备视频...";
                showUploadModal.value = false;

                // 本地预览直接指向服务器已开放的 /video 静态分发口
                videoUrl.value = window.location.origin + "/video/" + fileName;
                imageUrl.value = null;
                resetCurrentSession();

                try {
                    statusMessage.value = "正在提取视频底层特征及建立时空图谱...";
                    const startRes = await axios.post(`${API_BASE}/video/start_session`, {
                        video_path: "video/" + fileName
                    });

                    sessionId.value = startRes.data.session_id;
                    statusMessage.value = "极速通道：追踪实例建立完毕";
                } catch (e) {
                    console.error("Local Video Binding Failed: ", e);
                    statusMessage.value = "本地视频建立失败，请检查文件是否存在";
                } finally {
                    isLoading.value = false;
                }
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

                // 深度清空历史遗单
                annotations.value = [];
                maskDataCache.clear();
                lastGeneratedMask.value = '';
                rawMultiMasksB64.value = [];
                lastMultiMasksB64.value = [];
                textPrompt.value = '';
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

            // 放开识别模式下的交互选取
            if (!sessionId.value || isLoading.value) return;
            if (!['labeling', 'recognition', 'tracking'].includes(subTab.value)) return;

            // [取消自动清空] 允许用户利用点选/框选来优化调整之前的文字识别结果
            // if (textPrompt.value) {
            //     textPrompt.value = '';
            // }

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
            // 【改 2】：放开识别模式下的交互选取
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
                            text_threshold: parseFloat(confidenceThreshold.value)
                        });
                        maskBase64 = res.data.mask_base64;
                        rawMultiMasksB64.value = res.data.multi_masks_base64 || [];
                        lastGeneratedMask.value = maskBase64;
                        // 这里我们获取到了 raw 后，执行 IOU 过滤
                        await applyIouFilter();
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

        const startPolling = () => {
            if (!fetchTasksInterval) {
                fetchTasksInterval = setInterval(fetchTasks, 2000);
            }
        };

        const stopPolling = () => {
            if (fetchTasksInterval) {
                clearInterval(fetchTasksInterval);
                fetchTasksInterval = null;
            }
        };

        // 按需响应：只有处于 tracking 并且存在相关任务时，才启动轮询器
        watchEffect(() => {
            const hasActive = trackingTasks.value.some(t => t.status === 'processing');
            if (subTab.value === 'tracking' && hasActive) {
                startPolling();
            } else {
                stopPolling();
            }
        });

        const stopOrDeleteTask = async (sid) => {
            try {
                await axios.delete(`${API_BASE}/video/tasks/${sid}`);
                fetchTasks();
            } catch (e) {
                console.error("Delete task failed", e);
            }
        };

        const previewTrackingResult = (url) => {
            trackingResultUrl.value = url + "?t=" + Date.now();
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

                // 为了立即激活 watchEffect 的轮询钩子，强行写入一个伪处理态
                trackingTasks.value.push({
                    session_id: sessionId.value,
                    status: 'processing',
                    progress: 0,
                    totalFrames: 0
                });
                startPolling(); // 确保探测齿轮立即工作

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
                    similarity_threshold: parseFloat(confidenceThreshold.value),
                    text_threshold: parseFloat(confidenceThreshold.value)
                });

                if (res.data.mask_base64) {
                    lastGeneratedMask.value = res.data.mask_base64;
                    rawMultiMasksB64.value = res.data.multi_masks_base64 || [];
                    await applyIouFilter();
                }
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

            // 2. Multi-masks (待确认池)
            if (lastMultiMasksB64.value && lastMultiMasksB64.value.length > 0) {
                lastMultiMasksB64.value.forEach(item => {
                    imagesToLoad.push({ src: item.mask_base64, type: 'candidate' });
                });
            }

            // [修复显示遮挡] 即使有待确认池，单一生成的交互掩码(lastGeneratedMask)也应显示，用于即时反馈
            if (lastGeneratedMask.value) {
                imagesToLoad.push({ src: lastGeneratedMask.value, type: 'current_mask' });
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
                // Draw candidate multi-masks
                const candidates = imagesToLoad.filter(i => i.type === 'candidate');

                candidates.forEach((current, idx) => {
                    if (current && current.img) {
                        if (hoveredPendingIdx.value === idx) {
                            // 悬停时：完全不透明展示，附加强烈发光，并绘制两次以显著加深原本半透明的面罩颜色
                            ctx.globalAlpha = 1.0;
                            ctx.shadowColor = getTagColor(selectedTagId.value);
                            ctx.shadowBlur = 20;
                            ctx.drawImage(current.img, 0, 0);
                            ctx.shadowBlur = 0; // 第二次绘制不需要阴影叠加，纯粹加深色块
                            ctx.drawImage(current.img, 0, 0);
                        } else {
                            // 未悬停时：进一步降低透明度以凸显对比
                            ctx.globalAlpha = 0.4;
                            ctx.shadowBlur = 0;
                            ctx.drawImage(current.img, 0, 0);
                        }
                        ctx.globalAlpha = 1.0;
                    }
                });

                // Draw current active mask (from lastGeneratedMask)
                const activeOnes = imagesToLoad.filter(i => i.type === 'current_mask');
                activeOnes.forEach(current => {
                    if (current && current.img) {
                        ctx.globalAlpha = 0.8;
                        ctx.drawImage(current.img, 0, 0);
                        ctx.globalAlpha = 1.0;
                    }
                });

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
            rawMultiMasksB64.value = [];
            lastMultiMasksB64.value = []; // Reset multi-masks as well
            recognitionResult.value = null;
            textPrompt.value = ''; // 核心修复：彻底清空残余的文字搜索条件

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
            imageUrl, videoUrl, trackingResultUrl, sessionId, statusMessage, isLoading,
            isLeftPanelExpanded, isRightPanelExpanded,
            tags, selectedTagId, isCreatingTag, newTagName, newTagColor, annotations, hoveredAnnId, hoveredPendingIdx,
            zoomLevel, panOffset, globalSpacePushed, isPanDragging, isHintExpanded,
            lastGeneratedMask, lastMultiMasksB64, recognitionResult, currentFrameIdx, textPrompt,
            handleWheel, startPan, doPan, endPan,
            onImageSelected, onStreamFrame, handleUploadFromComponent, handleLocalVideo, handleVideoLoaded, onImageLoaded, saveCurrentAnnotation, confirmMultiTargets, confirmSingleTarget, cancelSingleTarget, clearPending, resetCurrentSession,
            handleMouseDown, handleMouseMove, handleMouseUp, requestPrediction, requestRecognition, requestSimilarSeg, startVideoTracking,
            confirmCreateTag, deleteTag, toggleTagVisibility, getTagColor, getTagName, deleteAnnotation,
            handleUpload, handleTextPromptSubmit, confidenceThreshold, applyIouFilter, handleThresholdChange,
            points, currentBox, dragBox, showTrackingResult, iouThreshold,
            trackingTasks, stopOrDeleteTask, previewTrackingResult
        };
    }
};
