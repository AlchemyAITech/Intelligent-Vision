import { ref, onMounted, onUnmounted, watch } from 'vue';

export default {
    name: 'ImageSource',
    emits: ['image-selected', 'stream-frame'],
    props: {
        hideCaptureBtn: {
            type: Boolean,
            default: false
        },
        hideVideoPreview: {
            type: Boolean,
            default: false
        },
        defaultSourceType: {
            type: String,
            default: 'upload'
        },
        autoStart: {
            type: Boolean,
            default: false
        },
        minimal: {
            type: Boolean,
            default: false
        }
    },
    template: `
        <div class="image-source-selector">
            <div class="control-group" v-if="!minimal">
                <label>选择输入源</label>
                <div class="radio-group" style="display:flex; flex-direction:column; gap:5px; margin-bottom:10px;">
                    <label><input type="radio" v-model="sourceType" value="upload"> 🖼️ 图片上传</label>
                    <label><input type="radio" v-model="sourceType" value="video"> 🎦 视频流分析</label>
                    <label><input type="radio" v-model="sourceType" value="local"> 📂 本地文件</label>
                    <label><input type="radio" v-model="sourceType" value="webcam"> 📷 摄像头</label>
                </div>
            </div>

            <!-- Upload -->
            <div v-show="sourceType === 'upload'" class="source-panel">
                <div style="border: 2px dashed rgba(162, 28, 175, 0.3); padding: 20px; text-align: center; border-radius: 8px; background: rgba(255,255,255,0.5);">
                    <input type="file" id="file-upload-input" @change="handleFileUpload" accept="image/*" style="display: none;">
                    <label for="file-upload-input" class="btn-primary" style="display:inline-block; margin-bottom: 10px; cursor:pointer;">
                        点此选择图片
                    </label>
                    <div v-if="selectedFileName" style="font-size: 13px; color: var(--primary-accent); word-break: break-all;">
                        📄 {{ selectedFileName }}
                    </div>
                    <div v-else style="font-size: 13px; color: var(--text-muted);">
                        支持 JPG, PNG 等格式
                    </div>
                </div>
            </div>

            <!-- Video -->
            <div v-show="sourceType === 'video'" class="source-panel">
                <div style="border: 2px dashed rgba(162, 28, 175, 0.3); padding: 20px; text-align: center; border-radius: 8px; background: rgba(255,255,255,0.5);">
                    <input type="file" id="video-upload-input" @change="handleVideoUpload" accept="video/mp4,video/webm,video/ogg" style="display: none;">
                    <label for="video-upload-input" class="btn-primary" style="display:inline-block; margin-bottom: 10px; cursor:pointer;">
                        点此加载本地视频
                    </label>
                    <div v-if="selectedVideoName" style="font-size: 13px; color: var(--primary-accent); word-break: break-all;">
                        🎥 {{ selectedVideoName }}
                    </div>
                    <div v-else style="font-size: 13px; color: var(--text-muted);">
                        支持 MP4, WebM 等视频格式 (免联机秒解封)
                    </div>
                </div>
                
                <div v-if="videoUrl" style="margin-top:10px;">
                    <video ref="uploadedVideoEl" :src="videoUrl" controls style="width:100%; border-radius:4px; max-height:200px;" @play="onVideoPlay" @pause="onVideoPause" @ended="onVideoPause" @seeked="onVideoSeeked" v-show="!hideVideoPreview"></video>
                    <p style="font-size:12px; color:#666; margin-top:5px;" v-if="hideVideoPreview">💡 视频播放已自动内嵌至分析台，请留意右侧画布。原生控制面板已隐藏。</p>
                    <p style="font-size:12px; color:#666; margin-top:5px;" v-else>💡 操作左侧播放器，右侧大屏将同步渲染实时分析流！</p>
                </div>
            </div>

            <!-- Local -->
            <div v-show="sourceType === 'local'" class="source-panel">
                <select v-model="selectedLocalFile" @change="handleLocalSelect" style="width:100%; padding:8px;">
                    <option value="" disabled>--选择已缓存文件--</option>
                    <option v-for="file in localFiles" :key="file" :value="file">{{ file }}</option>
                </select>
                <div v-if="localFiles.length===0" style="color:#888; font-size:12px; margin-top:5px;">暂无图片</div>
            </div>

            <!-- Webcam -->
            <div v-show="sourceType === 'webcam'" class="source-panel">
                <div v-if="!isWebcamActive">
                    <button class="btn-primary" @click="startWebcam" style="width:100%">启动摄像头</button>
                </div>
                <div v-else>
                    <video ref="videoEl" autoplay playsinline style="width:100%; border-radius:4px; transform: scaleX(-1); display:block;" v-show="!hideVideoPreview"></video>
                    <button v-if="!hideCaptureBtn" class="btn-primary" @click="captureWebcam" style="width:100%; margin-top:10px;">拍照并应用</button>
                    <button v-if="!minimal" class="btn-secondary" @click="stopWebcam" style="width:100%; margin-top:5px;">关闭摄像头</button>
                </div>
            </div>
            <canvas ref="canvasEl" style="display:none;"></canvas>
        </div>
    `,
    setup(props, { emit }) {
        const sourceType = ref(props.defaultSourceType || 'upload');
        const localFiles = ref([]);
        const selectedLocalFile = ref('');
        const selectedFileName = ref('');

        const selectedVideoName = ref('');
        const videoUrl = ref('');
        const uploadedVideoEl = ref(null);
        let videoInterval = null;

        const isWebcamActive = ref(false);
        const videoEl = ref(null);
        const canvasEl = ref(null);
        let stream = null;

        onMounted(async () => {
            try {
                const res = await axios.get('/api/common/local_images');
                localFiles.value = res.data.files || [];
            } catch (e) {
                console.error("Failed to load local images", e);
            }
            // Auto start if requested
            if (props.autoStart && sourceType.value === 'webcam') {
                startWebcam();
            }
        });

        onUnmounted(() => {
            stopWebcam();
            if (videoUrl.value) URL.revokeObjectURL(videoUrl.value);
            if (videoInterval) clearInterval(videoInterval);
        });

        watch(sourceType, (newVal) => {
            if (newVal !== 'video') {
                if (uploadedVideoEl.value) {
                    uploadedVideoEl.value.pause();
                }
            }
            if (newVal !== 'webcam') {
                stopWebcam();
            }
        });

        const handleFileUpload = (event) => {
            const file = event.target.files[0];
            if (file) {
                selectedFileName.value = file.name;
                emit('image-selected', { type: 'file', data: file });
            }
        };

        const handleLocalSelect = async () => {
            if (!selectedLocalFile.value) return;
            try {
                // Fetch the image as blob through axios to respect baseURL Config for cross-origin local testing
                const response = await axios.get('/images/' + selectedLocalFile.value, { responseType: 'blob' });
                const blob = response.data;
                const file = new File([blob], selectedLocalFile.value, { type: blob.type });
                emit('image-selected', { type: 'file', data: file });
            } catch (e) {
                console.error("Failed to load local image blob", e);
            }
        };

        const handleVideoUpload = (event) => {
            const file = event.target.files[0];
            if (file) {
                selectedVideoName.value = file.name;
                if (videoUrl.value) URL.revokeObjectURL(videoUrl.value);
                videoUrl.value = URL.createObjectURL(file);
            }
        };

        const emitVideoFrame = () => {
            const video = uploadedVideoEl.value;
            const canvas = canvasEl.value;
            if (!canvas || !video || !video.videoWidth) return;
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            // Compress moderately so backend parsing stays fast
            const b64 = canvas.toDataURL('image/jpeg', 0.65);
            emit('stream-frame', b64);
        };

        const onVideoPlay = () => {
            if (videoInterval) clearInterval(videoInterval);
            videoInterval = setInterval(() => {
                if (uploadedVideoEl.value && !uploadedVideoEl.value.paused && !uploadedVideoEl.value.ended) {
                    emitVideoFrame();
                }
            }, 66); // 15 FPS transmission rate
        };

        const onVideoPause = () => {
            if (videoInterval) {
                clearInterval(videoInterval);
                videoInterval = null;
            }
        };

        const onVideoSeeked = () => {
            // When seeking while paused, grab one single frame for immediate preview
            if (uploadedVideoEl.value && uploadedVideoEl.value.paused) {
                emitVideoFrame();
            }
        };

        const startWebcam = async () => {
            try {
                if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                    throw new Error("您的浏览器环境限制了摄像头访问。请尝试使用 http://localhost:8000 (或 127.0.0.1) 访问，或者检查浏览器权限。");
                }
                stream = await navigator.mediaDevices.getUserMedia({ video: true });
                isWebcamActive.value = true;
                setTimeout(() => {
                    if (videoEl.value) {
                        videoEl.value.srcObject = stream;
                    }
                }, 100);
            } catch (e) {
                alert("无法访问摄像头: " + e.message);
            }
        };

        const stopWebcam = () => {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }
            isWebcamActive.value = false;
        };

        const captureWebcam = () => {
            if (!videoEl.value || !canvasEl.value) return;
            const video = videoEl.value;
            const canvas = canvasEl.value;
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');

            // Mirror image if video is mirrored via CSS
            ctx.translate(canvas.width, 0);
            ctx.scale(-1, 1);
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            canvas.toBlob((blob) => {
                if (blob) {
                    const file = new File([blob], "webcam_capture.jpg", { type: "image/jpeg" });
                    emit('image-selected', { type: 'file', data: file });
                    stopWebcam();
                }
            }, 'image/jpeg');
        };

        // Real-time streaming logic
        let streamInterval = null;
        watch(isWebcamActive, (active) => {
            if (active) {
                streamInterval = setInterval(() => {
                    if (videoEl.value && videoEl.value.readyState >= 3) {
                        const video = videoEl.value;
                        const canvas = canvasEl.value;
                        if (!canvas) return;
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;
                        const ctx = canvas.getContext('2d');
                        ctx.translate(canvas.width, 0);
                        ctx.scale(-1, 1);
                        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                        const b64 = canvas.toDataURL('image/jpeg', 0.7);
                        emit('stream-frame', b64);
                    }
                }, 66); // 15 FPS
            } else {
                if (streamInterval) clearInterval(streamInterval);
            }
        });

        return {
            sourceType,
            localFiles,
            selectedLocalFile,
            selectedFileName,
            isWebcamActive,
            videoEl,
            canvasEl,
            selectedVideoName,
            videoUrl,
            uploadedVideoEl,
            handleVideoUpload,
            onVideoPlay,
            onVideoPause,
            onVideoSeeked,
            handleFileUpload,
            handleLocalSelect,
            startWebcam,
            stopWebcam,
            captureWebcam
        };
    }
}
