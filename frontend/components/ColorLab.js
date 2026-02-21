import { ref } from 'vue';
import ImageSource from './ImageSource.js';

export default {
    name: 'ColorLab',
    components: {
        ImageSource
    },
    template: `
    <div class="color-lab">
        <h2>🎨 色彩空间实验室</h2>
        <div class="layout-grid">
            
            <div class="sidebar-panel">
                <h3>1. 图像源</h3>
                <ImageSource @image-selected="onImageSelected" />
                <hr>
                
                <h3>2. 分析设置</h3>
                <div class="control-group" v-if="imageUrl">
                    <label>目标色彩空间</label>
                    <select v-model="targetSpace" @change="requestAnalysis">
                        <option value="RGB (Original)">RGB (Original)</option>
                        <option value="Grayscale (L)">Grayscale (L)</option>
                        <option value="YCbCr">YCbCr</option>
                        <option value="HSV">HSV</option>
                    </select>
                </div>
            </div>

            <div class="main-panel">
                <h3>3. 直观对比</h3>
                <div v-if="!imageUrl" class="empty-state">请加载一张图片。</div>
                <div v-else>
                    <div class="image-compare">
                        <div class="img-box">
                            <h4>原始图像 (RGB)</h4>
                            <img :src="imageUrl" class="preview-img">
                        </div>
                        <div class="img-box" v-if="resultImageUrl">
                            <h4>转换图像 ({{ targetSpace }})</h4>
                            <img :src="resultImageUrl" class="preview-img">
                        </div>
                    </div>
                    
                    <hr>
                    
                    <h3>📊 通道拆分与矩阵数据 (Center 20x20)</h3>
                    <div v-if="isLoading" class="loading-state">分析中...</div>
                    <div v-else-if="analysisResult" class="matrices-container">
                        <div v-for="(matrix, idx) in analysisResult.matrices" :key="idx" class="matrix-row">
                            <h4>Channel {{ idx + 1 }}: {{ analysisResult.channel_names[idx] }}</h4>
                            <div class="matrix-grid-wrapper">
                                <table class="matrix-table">
                                    <tr v-for="(row, rIdx) in matrix" :key="rIdx">
                                        <td v-for="(val, cIdx) in row" :key="cIdx" :style="getCellStyle(val)">
                                            {{ Math.round(val) }}
                                        </td>
                                    </tr>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
        </div>
    </div>
    `,
    setup() {
        const imageUrl = ref('');
        const selectedFile = ref(null);
        const targetSpace = ref('RGB (Original)');
        const resultImageUrl = ref('');
        const analysisResult = ref(null);
        const isLoading = ref(false);

        const onImageSelected = (payload) => {
            if (payload && payload.data) {
                selectedFile.value = payload.data;
                imageUrl.value = URL.createObjectURL(payload.data);
                requestAnalysis();
            }
        };

        const requestAnalysis = async () => {
            if (!selectedFile.value) return;
            isLoading.value = true;

            const formData = new FormData();
            formData.append('file', selectedFile.value);
            formData.append('target_space', targetSpace.value);

            try {
                const res = await axios.post('/api/image/color_space', formData);
                resultImageUrl.value = res.data.image_b64;
                analysisResult.value = {
                    matrices: res.data.matrices,
                    channel_names: res.data.channel_names
                };
            } catch (err) {
                console.error("Color analysis failed:", err);
                alert("色彩空间分析失败: " + (err.response?.data?.detail || err.message));
            } finally {
                isLoading.value = false;
            }
        };

        const getCellStyle = (val) => {
            // value is 0-255. Map to grayscale background
            const lum = val;
            const fg = lum > 128 ? '#000' : '#fff';
            return {
                backgroundColor: 'rgb(' + lum + ', ' + lum + ', ' + lum + ')',
                color: fg
            };
        };

        return {
            imageUrl,
            targetSpace,
            resultImageUrl,
            analysisResult,
            isLoading,
            onImageSelected,
            requestAnalysis,
            getCellStyle
        };
    }
}
