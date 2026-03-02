import { ref, onMounted } from 'vue';

const API_BASE = 'http://127.0.0.1:8000/api';

export default {
    name: 'DataManagement',
    template: `
    <div style="padding: 24px; height: 100%; display: flex; flex-direction: column; box-sizing: border-box;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px;">
            <h2 style="font-size: 24px; font-weight: bold; color: #82318E;">数据管理仓</h2>
            <div style="display: flex; gap: 12px;">
                <button :class="['nav-tab', activeTab === 'overview' ? 'active-tab' : 'inactive-tab']" @click="activeTab = 'overview'">📊 数据仓库列表</button>
                <button :class="['nav-tab', activeTab === 'classification' ? 'active-tab' : 'inactive-tab']" @click="activeTab = 'classification'">🏷️ 分类打标仓</button>
                <button :class="['nav-tab', activeTab === 'detection' ? 'active-tab' : 'inactive-tab']" @click="activeTab = 'detection'">🎯 框选与分割基站</button>
            </div>
        </div>
        
        <!-- Tab: Overview (Database Control Panel) -->
        <div v-if="activeTab === 'overview'" style="flex: 1; display: flex; flex-direction: column;">
            <div style="margin-bottom: 24px; display: flex; gap: 16px;">
                <button style="padding: 10px 20px; background: #82318E; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: bold; box-shadow: 0 4px 6px rgba(130,49,142,0.2);" @click="showCreateModal = true">+ 新建数据仓库</button>
                <button style="padding: 10px 20px; background: white; color: #82318E; border: 1px solid #82318E; border-radius: 8px; cursor: pointer; font-weight: bold;" @click="fetchDatabases">刷新仓库列表 🔄</button>
            </div>

            <!-- Databases Grid -->
            <div v-if="isLoading" style="text-align: center; margin-top: 50px; color: #718096;">正在加载数据仓库...</div>
            <div v-else-if="databases.length === 0" style="text-align: center; margin-top: 50px; color: #a0aec0; padding: 40px; background: white; border-radius: 12px; border: 1px dashed #cbd5e0;">
                <div style="font-size: 40px; margin-bottom: 12px;">🗂️</div>
                <div>尚无数据仓库，请点击上方按钮新建一个。</div>
            </div>
            <div v-else style="display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; flex: 1; align-content: start;">
                <div v-for="db in databases" :key="db.name" style="background: white; border-radius: 12px; padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; display: flex; flex-direction: column; transition: transform 0.2s; cursor: pointer;" class="db-card">
                    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 12px;">
                        <h3 style="font-size: 18px; font-weight: bold; color: #2d3748; margin: 0; display: flex; align-items: center; gap: 8px;">
                            <span style="color: #82318E;">📁</span> {{ db.name }}
                        </h3>
                        <button @click.stop="deleteDatabase(db.name)" style="background: none; border: none; color: #e53e3e; cursor: pointer; padding: 4px; border-radius: 4px;" title="删除仓库" class="delete-btn">
                            🗑️
                        </button>
                    </div>
                    <p style="color: #718096; font-size: 14px; flex: 1; margin: 0 0 16px 0; min-height: 40px; line-height: 1.5;">
                        {{ db.description || '暂无描述' }}
                    </p>
                    <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #edf2f7; padding-top: 12px; font-size: 13px; color: #a0aec0;">
                        <span>图像总数: <strong>{{ db.image_count }}</strong> 张</span>
                        <span style="color: #38a169; font-weight: bold;">就绪</span>
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

        <!-- Modal: Create Database -->
        <div v-if="showCreateModal" style="position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; z-index: 1000;">
            <div style="background: white; padding: 32px; border-radius: 12px; width: 400px; box-shadow: 0 10px 25px rgba(0,0,0,0.2);">
                <h3 style="margin-top: 0; margin-bottom: 20px; color: #2d3748;">新建数据仓库</h3>
                
                <div style="margin-bottom: 16px;">
                    <label style="display: block; font-size: 14px; font-weight: bold; margin-bottom: 8px; color: #4a5568;">数据仓名称 (仅限英文数字)</label>
                    <input type="text" v-model="newDbName" placeholder="例如: Medical_Dataset_v1" style="width: 100%; padding: 10px; border: 1px solid #e2e8f0; border-radius: 6px; box-sizing: border-box;" />
                </div>
                
                <div style="margin-bottom: 24px;">
                    <label style="display: block; font-size: 14px; font-weight: bold; margin-bottom: 8px; color: #4a5568;">描述 (可选)</label>
                    <textarea v-model="newDbDesc" placeholder="简要描述该数据集的用途或来源..." rows="3" style="width: 100%; padding: 10px; border: 1px solid #e2e8f0; border-radius: 6px; box-sizing: border-box; resize: none;"></textarea>
                </div>
                
                <div style="display: flex; justify-content: flex-end; gap: 12px;">
                    <button @click="showCreateModal = false" style="padding: 10px 20px; background: white; color: #4a5568; border: 1px solid #cbd5e0; border-radius: 6px; cursor: pointer; font-weight: bold;">取消</button>
                    <button @click="createDatabase" style="padding: 10px 20px; background: #82318E; color: white; border: none; border-radius: 6px; cursor: pointer; font-weight: bold;">确认创建</button>
                </div>
            </div>
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
            .db-card:hover {
                transform: translateY(-4px);
                border-color: #d6bcfa !important;
                box-shadow: 0 10px 15px -3px rgba(130,49,142, 0.1), 0 4px 6px -2px rgba(130,49,142, 0.05) !important;
            }
            .delete-btn:hover {
                background: #fed7d7 !important;
            }
        </style>
    </div>
    `,
    setup() {
        const activeTab = ref('overview');
        const selectedImages = ref([]);
        const currentTag = ref('Benign');

        // Database Management State
        const databases = ref([]);
        const isLoading = ref(false);
        const showCreateModal = ref(false);
        const newDbName = ref('');
        const newDbDesc = ref('');

        const fetchDatabases = async () => {
            isLoading.value = true;
            try {
                const res = await axios.get(`${API_BASE}/dataset/list`);
                if (res.data.status === 'success') {
                    databases.value = res.data.data;
                }
            } catch (err) {
                console.error("加载数据集列表失败:", err);
            } finally {
                isLoading.value = false;
            }
        };

        const createDatabase = async () => {
            if (!newDbName.value.trim()) {
                alert("请输入数据仓库名称！");
                return;
            }

            try {
                const res = await axios.post(`${API_BASE}/dataset/create`, {
                    project_name: newDbName.value.trim(),
                    description: newDbDesc.value.trim()
                });
                if (res.data.status === 'success') {
                    showCreateModal.value = false;
                    newDbName.value = '';
                    newDbDesc.value = '';
                    fetchDatabases(); // 刷新列表
                }
            } catch (err) {
                alert(`创建失败: ${err.response?.data?.detail || err.message}`);
            }
        };

        const deleteDatabase = async (dbName) => {
            if (!confirm(`警告：您确定要彻底删除数据仓库 [${dbName}] 吗？此操作不可逆！`)) {
                return;
            }
            try {
                const res = await axios.delete(`${API_BASE}/dataset/${dbName}`);
                if (res.data.status === 'success') {
                    fetchDatabases(); // 刷新列表
                }
            } catch (err) {
                alert(`删除失败: ${err.response?.data?.detail || err.message}`);
            }
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

        // Load databases on mount
        onMounted(() => {
            fetchDatabases();
        });

        return {
            activeTab,
            selectedImages,
            currentTag,

            databases,
            isLoading,
            showCreateModal,
            newDbName,
            newDbDesc,

            fetchDatabases,
            createDatabase,
            deleteDatabase,
            toggleSelectImage,
            applyTag
        };
    }
};
