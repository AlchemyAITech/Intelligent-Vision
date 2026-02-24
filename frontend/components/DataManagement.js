import { ref } from 'vue';

export default {
    name: 'DataManagement',
    template: `
    <div style="padding: 24px; height: 100%; display: flex; flex-direction: column; box-sizing: border-box;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px;">
            <h2 style="font-size: 24px; font-weight: bold; color: #82318E;">数据管理仓</h2>
            <div style="display: flex; gap: 12px;">
                <button :class="['nav-tab', activeTab === 'overview' ? 'active-tab' : 'inactive-tab']" @click="activeTab = 'overview'">📊 数据概览</button>
                <button :class="['nav-tab', activeTab === 'classification' ? 'active-tab' : 'inactive-tab']" @click="activeTab = 'classification'">🏷️ 分类打标仓</button>
                <button :class="['nav-tab', activeTab === 'detection' ? 'active-tab' : 'inactive-tab']" @click="activeTab = 'detection'">🎯 框选与分割基站</button>
            </div>
        </div>
        
        <!-- Tab: Overview -->
        <div v-if="activeTab === 'overview'" style="flex: 1; display: flex; flex-direction: column;">
            <div style="margin-bottom: 24px; display: flex; gap: 16px;">
                <button style="padding: 10px 20px; background: #82318E; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: bold; box-shadow: 0 4px 6px rgba(130,49,142,0.2);" @click="uploadFiles">上传本地图片/文件夹 📁</button>
                <button style="padding: 10px 20px; background: white; color: #82318E; border: 1px solid #82318E; border-radius: 8px; cursor: pointer; font-weight: bold;">刷新数据集 🔄</button>
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px; min-height: 400px;">
                <!-- 缩略图墙 -->
                <div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); border: 1px solid #f0e6f5;">
                    <h3 style="font-weight: bold; margin-bottom: 16px; color: #4a5568;">图片预览墙</h3>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; max-height: 300px; overflow-y: auto;">
                        <div v-for="i in 12" :key="i" style="aspect-ratio: 1; background: #f7fafc; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #a0aec0; font-size: 12px; border: 1px solid #e2e8f0;">
                            Image {{i}}
                        </div>
                    </div>
                    <div style="margin-top: 16px; font-size: 14px; color: #718096;">共计: 124 张影像</div>
                </div>

                <!-- 数据统计 -->
                <div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); border: 1px solid #f0e6f5;">
                    <h3 style="font-weight: bold; margin-bottom: 16px; color: #4a5568;">类别分布分析</h3>
                    <div style="display: flex; flex-direction: column; gap: 16px;">
                        <div>
                            <div style="display: flex; justify-content: space-between; font-size: 14px; margin-bottom: 4px;"><span>Benign (良性)</span><span>45%</span></div>
                            <div style="width: 100%; height: 8px; background: #edf2f7; border-radius: 4px; overflow: hidden;"><div style="width: 45%; height: 100%; background: #4299e1;"></div></div>
                        </div>
                        <div>
                            <div style="display: flex; justify-content: space-between; font-size: 14px; margin-bottom: 4px;"><span>Malignant (恶性)</span><span>30%</span></div>
                            <div style="width: 100%; height: 8px; background: #edf2f7; border-radius: 4px; overflow: hidden;"><div style="width: 30%; height: 100%; background: #f56565;"></div></div>
                        </div>
                        <div>
                            <div style="display: flex; justify-content: space-between; font-size: 14px; margin-bottom: 4px;"><span>Normal (正常)</span><span>25%</span></div>
                            <div style="width: 100%; height: 8px; background: #edf2f7; border-radius: 4px; overflow: hidden;"><div style="width: 25%; height: 100%; background: #48bb78;"></div></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Tab: Classification -->
        <div v-else-if="activeTab === 'classification'" style="flex: 1; display: flex; flex-direction: column; background: white; border-radius: 12px; padding: 20px; border: 1px solid #f0e6f5;">
            <h3 style="font-weight: bold; margin-bottom: 16px; color: #2d3748;">图片分门别类 (单签/多签)</h3>
            <div style="display: flex; gap: 20px; flex: 1; overflow: hidden;">
                <!-- 图片网格 -->
                <div style="flex: 3; display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 12px; overflow-y: auto; padding-right: 10px; align-content: start;">
                    <div v-for="i in 20" :key="i" style="aspect-ratio: 1; background: #edf2f7; border-radius: 8px; position: relative; border: 2px solid transparent; cursor: pointer;" :style="selectedImages.includes(i) ? 'border-color: #82318E; background: #faf5ff;' : ''" @click="toggleSelectImage(i)">
                        <div style="position: absolute; top: 4px; left: 4px; background: rgba(0,0,0,0.5); color: white; border-radius: 4px; padding: 2px 6px; font-size: 10px;">img_{{i}}.jpg</div>
                        <div v-if="selectedImages.includes(i)" style="position: absolute; top: 4px; right: 4px; background: #82318E; color: white; width: 20px; height: 20px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 12px;">✓</div>
                        <!-- Mock Label Badge -->
                        <div v-if="i % 3 === 0" style="position: absolute; bottom: 4px; right: 4px; background: #38a169; color: white; border-radius: 4px; padding: 2px 4px; font-size: 10px;">Benign</div>
                    </div>
                </div>
                <!-- 操作栏 -->
                <div style="flex: 1; min-width: 250px; border-left: 1px solid #e2e8f0; padding-left: 20px; display: flex; flex-direction: column;">
                    <div style="margin-bottom: 24px; font-size: 14px; color: #718096;">
                        已选中 <strong style="color: #82318E; font-size: 18px;">{{ selectedImages.length }}</strong> 张图像
                    </div>
                    <div style="margin-bottom: 16px;">
                        <label style="display: block; font-size: 14px; font-weight: bold; margin-bottom: 8px;">打上分类标签：</label>
                        <select v-model="currentTag" style="width: 100%; border: 1px solid #e2e8f0; padding: 10px; border-radius: 8px; margin-bottom: 16px;">
                            <option value="Benign">Benign (良性)</option>
                            <option value="Malignant">Malignant (恶性)</option>
                            <option value="Normal">Normal (正常)</option>
                        </select>
                        <button style="width: 100%; padding: 10px; background: #82318E; color: white; border: none; border-radius: 8px; font-weight: bold; cursor: pointer;" @click="applyTag">批量应用标签</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Tab: Detection -->
        <div v-else-if="activeTab === 'detection'" style="flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; background: white; border-radius: 12px; border: 1px dashed #cbd5e0;">
            <div style="font-size: 48px; margin-bottom: 16px;">🎯</div>
            <h3 style="font-weight: bold; color: #4a5568; margin-bottom: 8px;">通用检测 / 分割打标基站</h3>
            <p style="color: #a0aec0; text-align: center; max-width: 400px; margin-bottom: 24px;">此处将全量嵌入现有的 SAM 高维像素级分割接口与 YOLO Bounding Box 拉框模块，统一导出标准 txt 坐标系。</p>
            <button style="padding: 10px 24px; background: #edf2f7; color: #4a5568; font-weight: bold; border-radius: 8px; border: none; cursor: pointer;">正在研发载入协议...</button>
        </div>

        <style>
            .nav-tab {
                padding: 8px 16px;
                border-radius: 8px;
                font-weight: bold;
                border: 1px solid transparent;
                cursor: pointer;
                transition: all 0.2s;
            }
            .active-tab {
                background: #faf5ff;
                color: #82318E;
                border-color: #d6bcfa;
            }
            .inactive-tab {
                background: white;
                color: #a0aec0;
                border-color: #e2e8f0;
            }
            .inactive-tab:hover {
                background: #f7fafc;
                color: #718096;
            }
        </style>
    </div>
    `,
    setup() {
        const activeTab = ref('overview');
        const selectedImages = ref([]);
        const currentTag = ref('Benign');

        const uploadFiles = () => {
            alert('功能即将开放：批量上传 / 本地挂载目录');
        };

        const toggleSelectImage = (id) => {
            const idx = selectedImages.value.indexOf(id);
            if (idx === -1) selectedImages.value.push(id);
            else selectedImages.value.splice(idx, 1);
        };

        const applyTag = () => {
            if (selectedImages.value.length === 0) return;
            alert(`已为 ${selectedImages.value.length} 张图片成功应用分类范式：${currentTag.value}`);
            selectedImages.value = [];
        };

        return {
            activeTab,
            selectedImages,
            currentTag,
            uploadFiles,
            toggleSelectImage,
            applyTag
        };
    }
};
