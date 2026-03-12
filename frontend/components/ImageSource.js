import { ref, computed, onMounted, onUnmounted, watch } from 'vue';

export default {
    name: 'ImageSource',
    emits: ['image-selected', 'stream-frame', 'video-selected', 'local-video-selected'],
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
        disableStreaming: {
            type: Boolean,
            default: false
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
                <label style="font-size: 14px; font-weight: 600; margin-bottom: 12px; display: block;">选择输入源</label>
                <div class="radio-group" style="display:flex; flex-direction:column; gap:12px; margin-bottom:15px; padding: 10px; background: rgba(0,0,0,0.02); border-radius: 8px;">
                    <label style="padding: 5px; cursor: pointer; transition: 0.2s;"><input type="radio" v-model="sourceType" value="upload"> 🖼️ 图片上传</label>
                    <label style="padding: 5px; cursor: pointer; transition: 0.2s;"><input type="radio" v-model="sourceType" value="local"> 📂 本地图片缓存</label>
                    <label style="padding: 5px; cursor: pointer; transition: 0.2s;"><input type="radio" v-model="sourceType" value="local_video"> 🎞️ 本地视频缓存</label>
                    <label style="padding: 5px; cursor: pointer; transition: 0.2s;"><input type="radio" v-model="sourceType" value="webcam"> 📷 摄像头直出</label>
                </div>
            </div>

            <!-- Upload -->
            <div v-show="sourceType === 'upload'" class="source-panel">
                <div style="border: 2px dashed rgba(162, 28, 175, 0.4); padding: 40px 20px; text-align: center; border-radius: 12px; background: rgba(255,255,255,0.7); transition: all 0.3s;">
                    <input type="file" id="file-upload-input" @change="handleFileUpload" accept="image/*,video/*" style="display: none;">
                    <label for="file-upload-input" class="btn-primary" style="display:inline-block; margin-bottom: 15px; cursor:pointer; padding: 12px 25px; font-size: 15px;">
                        选取图/视频文件
                    </label>
                    <div v-if="selectedFileName" style="font-size: 14px; color: var(--primary-accent); word-break: break-all; font-weight: bold;">
                        成功加载: {{ selectedFileName }}
                    </div>
                    <div v-else style="font-size: 13px; color: var(--text-muted);">
                        支持常规图片及视频文件 (JPG、PNG、MP4 等)
                    </div>
                </div>
            </div>

            <!-- Local Image -->
            <div v-show="sourceType === 'local'" class="source-panel">
                <div style="display:flex; gap:10px;">
                    <select v-model="selectedLocalFile" @change="handleLocalSelect" style="flex:1; padding:8px;">
                        <option value="" disabled>--选择已缓存图片--</option>
                        <option v-for="file in localFiles" :key="file" :value="file">{{ file }}</option>
                    </select>
                    <button class="btn-primary" @click="handleLocalSelect" :disabled="!selectedLocalFile">加载</button>
                </div>
                <div v-if="localFiles.length===0" style="color:#888; font-size:12px; margin-top:5px;">暂无图片</div>
            </div>

            <!-- Local Video -->
            <div v-show="sourceType === 'local_video'" class="source-panel">
                <div style="display:flex; gap:10px;">
                    <select v-model="selectedLocalVideoFile" @change="handleLocalVideoSelect" style="flex:1; padding:8px;">
                        <option value="" disabled>--选择已缓存视频--</option>
                        <option v-for="file in localVideoFiles" :key="file" :value="file">{{ file }}</option>
                    </select>
                    <button class="btn-primary" @click="handleLocalVideoSelect" :disabled="!selectedLocalVideoFile">加载</button>
                </div>
                <div v-if="localVideoFiles.length===0" style="color:#888; font-size:12px; margin-top:5px;">暂无视频</div>
            </div>

            <!-- Video Preview (Webcam / Video Files) -->
            <div v-show="isVideoSource" class="source-panel">
                <div v-if="sourceType === 'webcam' && !isWebcamActive">
                    <button class="btn-primary" @click="startWebcam" style="width:100%">启动摄像头</button>
                </div>
                <div v-else>
                    <video ref="videoEl" autoplay playsinline loop muted :style="{ width:'100%', borderRadius:'4px', display:'block', transform: sourceType === 'webcam' ? 'scaleX(-1)' : 'none' }" v-show="!hideVideoPreview"></video>
                    
                    <button v-if="sourceType === 'webcam' && !hideCaptureBtn" class="btn-primary" @click="captureWebcam" style="width:100%; margin-top:10px;">拍照并应用</button>
                    <button v-if="sourceType === 'webcam' && !minimal" class="btn-secondary" @click="stopWebcam" style="width:100%; margin-top:5px;">关闭摄像头</button>
                    
                    <button v-if="sourceType !== 'webcam' && isVideoPlaying" class="btn-secondary" @click="stopVideo" style="width:100%; margin-top:5px;">停止推流</button>
                </div>
            </div>
            <canvas ref="canvasEl" style="display:none;"></canvas>
        </div>
    `,
    setup(props, { emit }) {

        const sourceType = ref(props.defaultSourceType || 'upload');
        const localFiles = ref([]);
        const localVideoFiles = ref([]);
        const selectedLocalFile = ref('');
        const selectedLocalVideoFile = ref('');
        const selectedFileName = ref('');

        const isWebcamActive = ref(false);
        const isUploadedVideo = ref(false);
        const isVideoPlaying = ref(false);

        const isVideoSource = computed(() => {
            return sourceType.value === 'webcam' || sourceType.value === 'local_video' || (sourceType.value === 'upload' && isUploadedVideo.value);
        });

        const videoEl = ref(null);
        const canvasEl = ref(null);
        let stream = null;
        let streamInterval = null;

        onMounted(async () => {
            try {
                const res = await axios.get('/api/common/local_images');
                localFiles.value = res.data.files || [];
            } catch (e) {
                console.error("Failed to load local images", e);
            }
            try {
                const resV = await axios.get('/api/common/local_videos');
                localVideoFiles.value = resV.data.files || [];
            } catch (e) {
                console.error("Failed to load local videos", e);
            }
            // Auto start if requested
            if (props.autoStart && sourceType.value === 'webcam') {
                startWebcam();
            }
        });

        onUnmounted(() => {
            stopWebcam();
            stopVideo();
        });

        watch(sourceType, (newVal) => {
            if (newVal !== 'webcam') stopWebcam();
            if (newVal !== 'local_video' && !(newVal === 'upload' && isUploadedVideo.value)) {
                stopVideo();
            }
        });

        const handleFileUpload = (event) => {
            const file = event.target.files[0];
            if (file) {
                selectedFileName.value = file.name;
                stopVideo();
                if (file.type.startsWith('video/')) {
                    isUploadedVideo.value = true;
                    playVideoSource(URL.createObjectURL(file));
                } else {
                    isUploadedVideo.value = false;
                    emit('image-selected', { type: 'file', data: file });
                }
            }
        };

        const handleLocalSelect = async () => {
            if (!selectedLocalFile.value) return;
            try {
                const response = await axios.get('/images/' + selectedLocalFile.value, { responseType: 'blob' });
                const blob = response.data;
                const file = new File([blob], selectedLocalFile.value, { type: blob.type });
                emit('image-selected', { type: 'file', data: file });
            } catch (e) {
                console.error("Failed to load local image blob", e);
            }
        };

        const handleLocalVideoSelect = () => {
            if (!selectedLocalVideoFile.value) return;
            selectedFileName.value = selectedLocalVideoFile.value;
            stopVideo();
            playVideoSource('/video/' + selectedLocalVideoFile.value);
            emit('local-video-selected', selectedLocalVideoFile.value);
        };

        const playVideoSource = (url) => {
            if (videoEl.value) {
                videoEl.value.src = url;
                videoEl.value.play().then(() => {
                    isVideoPlaying.value = true;
                    if (!props.disableStreaming) {
                        startStreamingLoop();
                    }
                }).catch(e => console.error("Video play failed", e));
            }
        };

        const stopVideo = () => {
            if (videoEl.value) {
                videoEl.value.pause();
                videoEl.value.removeAttribute('src');
                videoEl.value.load();
            }
            isVideoPlaying.value = false;
            stopStreamingLoop();
        };

        const startWebcam = async () => {
            try {
                if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                    throw new Error("您的浏览器环境限制了摄像头访问。请尝试使用 http://localhost:32100 访问。");
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

        const startStreamingLoop = () => {
            if (streamInterval) clearInterval(streamInterval);
            streamInterval = setInterval(() => {
                if (videoEl.value && videoEl.value.readyState >= 3 && !videoEl.value.paused) {
                    const video = videoEl.value;
                    const canvas = canvasEl.value;
                    if (!canvas) return;

                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    const ctx = canvas.getContext('2d');

                    if (sourceType.value === 'webcam') {
                        ctx.translate(canvas.width, 0);
                        ctx.scale(-1, 1);
                    }
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

                    const b64 = canvas.toDataURL('image/jpeg', 0.6); // 降低质量提高帧数传输性能
                    emit('stream-frame', b64);
                }
            }, 66); // ~15 FPS
        };

        const stopStreamingLoop = () => {
            if (streamInterval) {
                clearInterval(streamInterval);
                streamInterval = null;
            }
        };

        watch(isWebcamActive, (active) => {
            if (active) {
                startStreamingLoop();
            } else {
                stopStreamingLoop();
            }
        });

        return {
            sourceType,
            localFiles,
            localVideoFiles,
            selectedLocalFile,
            selectedLocalVideoFile,
            selectedFileName,
            isWebcamActive,
            isUploadedVideo,
            isVideoPlaying,
            isVideoSource,
            videoEl,
            canvasEl,
            handleFileUpload,
            handleLocalSelect,
            handleLocalVideoSelect,
            startWebcam,
            stopWebcam,
            captureWebcam,
            stopVideo
        };
    }
}
