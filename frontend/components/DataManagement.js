import { ref, onMounted, watch, computed } from 'vue';

const API_BASE = 'http://127.0.0.1:32100/api';

export default {
    name: 'DataManagement',
    template: `
    <div style="padding: 16px 24px; height: 100%; display: flex; flex-direction: column; box-sizing: border-box;">
        
        <!-- Header for List View -->
        <div v-if="currentView === 'list'" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px;">
            <h2 style="font-size: 24px; font-weight: bold; color: #82318E; margin: 0;">数据管理仓</h2>
        </div>
        
        <!-- Header for Detail View -->
        <div v-else-if="currentView === 'detail'" style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; flex-wrap: wrap; gap: 12px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <button @click="closeDataset" style="background: white; border: 1px solid #cbd5e0; border-radius: 6px; padding: 6px 12px; cursor: pointer; font-weight: bold; color: #4a5568; box-shadow: 0 1px 2px rgba(0,0,0,0.05); font-size: 13px;">
                    <span style="margin-right: 4px;">⬅️</span> 返回列表
                </button>
                <h2 style="font-size: 20px; font-weight: bold; color: #82318E; margin: 0; display: flex; align-items: center; gap: 6px;">📁 {{ selectedDataset?.name }}</h2>
            </div>
            
            <!-- Detail Level Tabs -->
            <div style="display: flex; gap: 4px; background: white; padding: 4px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid #e2e8f0;">
                <button :class="['nav-tab', detailTab === 'content' ? 'active-tab' : 'inactive-tab']" @click="detailTab = 'content'" style="padding: 6px 12px; border-radius: 6px; font-size: 13px;">📦 内容总览</button>
                <button :class="['nav-tab', detailTab === 'classification' ? 'active-tab' : 'inactive-tab']" @click="detailTab = 'classification'" style="padding: 6px 12px; border-radius: 6px; font-size: 13px;">🏷️ 分类打标仓</button>
                <button :class="['nav-tab', detailTab === 'detection' ? 'active-tab' : 'inactive-tab']" @click="detailTab = 'detection'" style="padding: 6px 12px; border-radius: 6px; font-size: 13px;">🎯 分割基站</button>
            </div>
        </div>

        <!-- Tab: Overview (Database Control Panel) -->
        <div v-if="currentView === 'list'" style="flex: 1; display: flex; flex-direction: column;">
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
                <div v-for="db in databases" :key="db.name" @click="openDataset(db)" style="background: white; border-radius: 12px; padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; display: flex; flex-direction: column; transition: transform 0.2s; cursor: pointer;" class="db-card">
                    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 12px;">
                        <h3 style="font-size: 18px; font-weight: bold; color: #2d3748; margin: 0; display: flex; align-items: center; gap: 8px;">
                            <span style="color: #82318E;">📁</span> {{ db.name }}
                        </h3>
                        <button @click.stop="databaseToDelete = db.name" style="background: none; border: none; color: #e53e3e; cursor: pointer; padding: 4px; border-radius: 4px;" title="删除仓库" class="delete-btn">
                            🗑️
                        </button>
                    </div>
                    <p style="color: #718096; font-size: 14px; flex: 1; margin: 0 0 16px 0; min-height: 40px; line-height: 1.5;">
                        {{ db.description || '暂无描述' }}
                    </p>
                    <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid #edf2f7; padding-top: 12px; font-size: 13px; color: #a0aec0;">
                        <span>图像总数: <strong>{{ db.image_count }}</strong> 张</span>
                        <span style="color: #38a169; font-weight: bold;">进入仓库 ➡</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- DETAIL VIEW Content Area -->
        <div v-else-if="currentView === 'detail'" style="flex: 1; display: flex; flex-direction: column; overflow: hidden;">
            
            <!-- Detail Tab: Content Overview -->
            <div v-if="detailTab === 'content'" style="flex: 1; display: flex; gap: 24px; padding: 16px; min-height: 0; background: white; border-radius: 12px; border: 1px solid #e2e8f0; overflow: hidden;">
                <!-- LEFT PANEL: IMAGE GRID AND CRUD -->
                <div style="flex: 3; display: flex; flex-direction: column; overflow: hidden;">
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; margin-bottom: 12px;">
                        <h4 style="margin: 0; color: #2d3748; font-size: 18px; display: flex; align-items: center; gap: 8px;">
                            <span style="color: #82318E;">🖼️</span> 样本浏览与管理
                        </h4>
                        
                        <div style="display: flex; gap: 12px;">
                            <input type="file" multiple accept="image/*" ref="fileInput" style="display: none" @change="handleFileUpload" />
                            <button @click="$refs.fileInput.click()" style="padding: 6px 16px; background: white; color: #38a169; border: 1px solid #38a169; border-radius: 6px; font-weight: bold; cursor: pointer; transition: all 0.2s; font-size: 13px;">
                                + 上传新增样本
                            </button>
                            <button @click="deleteSelectedImages" style="padding: 6px 16px; background: #fff5f5; color: #e53e3e; border: 1px solid #feb2b2; border-radius: 6px; font-weight: bold; cursor: pointer; transition: opacity 0.2s; font-size: 13px;" :style="selectedImages.length === 0 ? 'opacity: 0.5; cursor: not-allowed;' : ''">
                                删除选定样本
                            </button>
                        </div>
                    </div>

                    <!-- OVERVIEW Filters -->
                    <div style="display: flex; gap: 12px; align-items: center; background: #fdf2f8; padding: 10px 16px; border-radius: 8px; border: 1px dashed #fbcfe8; margin-bottom: 12px;">
                        <div style="display: flex; align-items: center; gap: 8px; flex-shrink: 0;">
                            <span style="font-size: 13px; color: #702459; font-weight: bold; white-space: nowrap;">检视版本:</span>
                            <select v-model="currentVersion" @change="resetPaginationAndFetch" style="padding: 4px 8px; border: 1px solid #cbd5e0; border-radius: 4px; outline: none; background: white; color: #4a5568; max-width: 150px; font-size: 13px;">
                                <option v-for="v in versions" :key="v.version" :value="v.version">{{ v.version }}</option>
                            </select>
                        </div>
                        <input type="text" v-model="searchQuery" @keyup.enter="resetPaginationAndFetch" placeholder="搜索文件名或包含的标签..." style="flex: 1; padding: 6px 12px; border: 1px solid #e2e8f0; border-radius: 4px; outline: none; font-size: 13px;" />
                        <button @click="resetPaginationAndFetch" style="padding: 6px 16px; background: #702459; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 13px;">搜索内容</button>
                    </div>

                    <!-- IMAGE GRID -->
                    <div v-if="isImagesLoading" style="text-align: center; padding: 40px; color: #a0aec0; flex: 1;">正在加载数据...</div>
                    <div v-else-if="paginatedImages.length === 0" style="text-align: center; padding: 40px; color: #a0aec0; background: #f7fafc; border-radius: 8px; border: 1px dashed #cbd5e0; flex: 1;">暂无匹配的图像数据</div>
                    
                    <div v-else style="display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 10px; overflow-y: auto; flex: 1; padding-right: 8px; align-content: start;">
                        <div v-for="img in paginatedImages" :key="img.name" style="aspect-ratio: 1; background: #edf2f7; border-radius: 6px; position: relative; border: 2px solid transparent; cursor: pointer; transition: all 0.2s; overflow: hidden;" :style="selectedImages.includes(img.name) ? 'border-color: #82318E; box-shadow: 0 0 0 2px rgba(130,49,142,0.2); transform: scale(0.96);' : ''" @click="toggleSelectImage(img.name)" @dblclick="openPreviewImage(img.name)">
                            <img :src="getImageUrl(img.name)" style="width: 100%; height: 100%; object-fit: cover; display: block;" loading="lazy" />
                            <div style="position: absolute; top: 0; left: 0; right: 0; background: linear-gradient(to bottom, rgba(0,0,0,0.6) 0%, transparent 100%); padding: 4px 6px 16px; pointer-events: none;">
                                <div style="color: white; font-size: 10px; text-shadow: 0 1px 2px rgba(0,0,0,0.8); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" :title="img.name">{{ img.name }}</div>
                            </div>
                            <div v-if="selectedImages.includes(img.name)" style="position: absolute; top: 4px; right: 4px; background: #82318E; color: white; width: 18px; height: 18px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">✓</div>
                        </div>
                    </div>

                    <!-- PAGINATION -->
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px; padding-top: 12px; border-top: 1px solid #e2e8f0;">
                        <div style="font-size: 13px; color: #718096; display: flex; gap: 12px; align-items: center;">
                            <span>共 <strong>{{ totalImages }}</strong> 项</span>
                            <span>已选 <strong style="color: #e53e3e;">{{ selectedImages.length }}</strong> 项</span>
                            <button @click="selectAllVisible" style="background: none; border: none; color: #3182ce; cursor: pointer; font-size: 12px; padding: 0;">全选本页</button>
                            <button v-if="selectedImages.length > 0" @click="selectedImages = []" style="background: none; border: none; color: #e53e3e; cursor: pointer; font-size: 12px; padding: 0;">清空</button>
                        </div>
                        <div style="display: flex; gap: 8px; align-items: center;">
                            <button :disabled="currentPage <= 1" @click="changePage(currentPage - 1)" style="padding: 4px 10px; border: 1px solid #e2e8f0; background: white; border-radius: 4px; cursor: pointer; font-size: 12px;" :style="currentPage <= 1 ? 'opacity: 0.5; cursor: not-allowed;' : ''">上一页</button>
                            <span style="font-size: 13px; color: #4a5568;">第 {{ currentPage }} / {{ Math.max(1, Math.ceil(totalImages / pageSize)) }} 页</span>
                            <button :disabled="currentPage >= Math.ceil(totalImages / pageSize)" @click="changePage(currentPage + 1)" style="padding: 4px 10px; border: 1px solid #e2e8f0; background: white; border-radius: 4px; cursor: pointer; font-size: 12px;" :style="currentPage >= Math.ceil(totalImages / pageSize) ? 'opacity: 0.5; cursor: not-allowed;' : ''">下一页</button>
                        </div>
                    </div>
                </div>

                <!-- RIGHT PANEL: STATS CARDS -->
                <div style="flex: 1; min-width: 250px; max-width: 320px; display: flex; flex-direction: column; gap: 16px; overflow-y: auto; border-left: 1px solid #e2e8f0; padding-left: 20px;">
                    <div style="display: flex; flex-direction: column; align-items: center; padding: 24px 0; border-bottom: 2px dashed #edf2f7; margin-bottom: 8px;">
                        <div style="font-size: 40px; margin-bottom: 8px;">📦</div>
                        <h3 style="font-weight: bold; color: #4a5568; margin-bottom: 4px;">仓库内容总览</h3>
                        <div style="color: #718096; font-size: 14px;">共 <strong style="color: #82318E; font-size: 20px;">{{ datasetStats.total }}</strong> 张资源</div>
                    </div>
                    
                    <div style="background: #f7fafc; padding: 16px; border-radius: 12px; border: 1px solid #edf2f7;">
                        <h4 style="margin: 0 0 16px; color: #2b6cb0; border-bottom: 2px solid #2b6cb0; padding-bottom: 8px; display: inline-block; font-size: 14px;">分集状态结构</h4>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 13px;"><span style="color: #4a5568;">训练集 (Train)</span><strong style="color: #2b6cb0;">{{ datasetStats.splits.train }}</strong></div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 13px;"><span style="color: #4a5568;">验证集 (Val)</span><strong style="color: #38a169;">{{ datasetStats.splits.val }}</strong></div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 13px;"><span style="color: #4a5568;">测试集 (Test)</span><strong style="color: #ed8936;">{{ datasetStats.splits.test }}</strong></div>
                        <div style="display: flex; justify-content: space-between; margin-top: 12px; padding-top: 12px; border-top: 1px dashed #cbd5e0; font-size: 13px;"><span style="color: #a0aec0;">未分配 (Unassigned)</span><strong style="color: #718096;">{{ datasetStats.splits.unassigned }}</strong></div>
                    </div>
                    
                    <div style="background: #faf5ff; padding: 16px; border-radius: 12px; border: 1px solid #e9d8fd;">
                        <h4 style="margin: 0 0 16px; color: #82318E; border-bottom: 2px solid #82318E; padding-bottom: 8px; display: inline-block; font-size: 14px;">标注进度墙 (V版本: {{ currentVersion }})</h4>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px; align-items: center; font-size: 13px;">
                            <span style="color: #4a5568;">已标注图像数</span>
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <div style="width: 80px; height: 6px; background: #e2e8f0; border-radius: 3px; overflow: hidden;">
                                    <div :style="'height: 100%; background: #82318E; width: ' + (datasetStats.total ? (datasetStats.annotated / datasetStats.total * 100) : 0) + '%'"></div>
                                </div>
                                <strong style="color: #82318E;">{{ datasetStats.annotated }}</strong>
                            </div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 13px;"><span style="color: #a0aec0;">未标注图像数</span><strong style="color: #718096;">{{ datasetStats.unannotated }}</strong></div>
                        <div style="display: flex; justify-content: space-between; margin-top: 12px; padding-top: 12px; border-top: 1px dashed #d6bcfa; font-size: 13px;"><span style="color: #4a5568;">类别池总数</span><strong style="color: #82318E;">{{ categories.length }}</strong>个</div>
                    </div>
                </div>

            </div>
            
            <!-- Detail Tab: Classification -->
            <div v-else-if="detailTab === 'classification'" style="flex: 1; display: flex; flex-direction: column; background: white; border-radius: 10px; padding: 16px; border: 1px solid #f0e6f5; overflow: hidden;">
                <!-- TOP CONTROL BAR -->
                <div style="display: flex; gap: 12px; margin-bottom: 12px; align-items: center; flex-wrap: wrap; background: #faf5ff; padding: 10px; border-radius: 8px; border: 1px solid #e9d8fd;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-weight: bold; color: #553c9a; white-space: nowrap;">标注库版本:</span>
                        <select v-model="currentVersion" @change="resetPaginationAndFetch" style="padding: 6px; border: 1px solid #d6bcfa; border-radius: 4px; outline: none; background: white; color: #4a5568; max-width: 200px;">
                            <option v-for="v in versions" :key="v.version" :value="v.version">
                                {{ v.version }} {{ v.description ? ' - ' + v.description : '' }}
                            </option>
                        </select>
                        <button @click="showCreateVersionModal = true" style="padding: 6px 10px; background: white; border: 1px dashed #b794f4; border-radius: 4px; color: #6b46c1; cursor: pointer; font-size: 13px;">+ 新版本</button>
                    </div>
                    
                    <div style="width: 1px; height: 24px; background: #d6bcfa; margin: 0 4px;"></div>
                    
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="color: #4a5568; font-size: 14px; white-space: nowrap;">分集筛选:</span>
                        <select v-model="filterSplit" @change="resetPaginationAndFetch" style="padding: 6px; border: 1px solid #e2e8f0; border-radius: 4px; outline: none;">
                            <option value="all">全部 (All)</option>
                            <option value="unassigned">未分配 (Unassigned)</option>
                            <option value="train">训练集 (Train)</option>
                            <option value="val">验证集 (Val)</option>
                            <option value="test">测试集 (Test)</option>
                        </select>
                    </div>

                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="color: #4a5568; font-size: 14px; white-space: nowrap;">状态筛选:</span>
                        <select v-model="filterStatus" @change="resetPaginationAndFetch" style="padding: 6px; border: 1px solid #e2e8f0; border-radius: 4px; outline: none;">
                            <option value="all">全部 (All)</option>
                            <option value="annotated">已打标 (Annotated)</option>
                            <option value="unannotated">未打标 (Unannotated)</option>
                        </select>
                    </div>

                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="color: #4a5568; font-size: 14px; white-space: nowrap;">标签筛选:</span>
                        <select v-model="filterCategory" @change="resetPaginationAndFetch" style="padding: 6px; border: 1px solid #e2e8f0; border-radius: 4px; outline: none; max-width: 150px;">
                            <option value="all">全部类别 (All)</option>
                            <option v-for="cat in categories" :key="cat" :value="cat">{{ cat }}</option>
                        </select>
                    </div>

                    <div style="flex: 1; display: flex; justify-content: flex-end;">
                        <input type="text" v-model="searchQuery" @keyup.enter="resetPaginationAndFetch" placeholder="搜索文件名或标签类别..." style="padding: 6px 12px; border: 1px solid #e2e8f0; border-radius: 4px; width: 100%; max-width: 200px; outline: none;" />
                        <button @click="resetPaginationAndFetch" style="padding: 6px 12px; background: #82318E; color: white; border: none; border-radius: 4px; margin-left: 8px; cursor: pointer;">搜索</button>
                    </div>
                </div>

                <div style="display: flex; gap: 20px; flex: 1; overflow: hidden;">
                    <!-- IMAGE GRID -->
                    <div style="flex: 3; display: flex; flex-direction: column; overflow: hidden;">
                        <div v-if="isImagesLoading" style="text-align: center; padding: 40px; color: #a0aec0;">正在加载数据...</div>
                        <div v-else-if="paginatedImages.length === 0" style="text-align: center; padding: 40px; color: #a0aec0;">未检索到匹配的图像数据</div>
                        
                        <div v-else style="display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 12px; overflow-y: auto; padding-right: 10px; align-content: start; flex: 1;">
                            <div v-for="img in paginatedImages" :key="img.name" style="aspect-ratio: 1; background: #edf2f7; border-radius: 8px; position: relative; border: 2px solid transparent; cursor: pointer; transition: all 0.2s; overflow: hidden;" :style="selectedImages.includes(img.name) ? 'border-color: #82318E; box-shadow: 0 0 0 2px rgba(130,49,142,0.2); transform: scale(0.96);' : ''" @click="toggleSelectImage(img.name)" @dblclick="openPreviewImage(img.name)">
                                
                                <img :src="getImageUrl(img.name)" style="width: 100%; height: 100%; object-fit: cover; display: block;" loading="lazy" />

                                <div style="position: absolute; top: 0; left: 0; right: 0; background: linear-gradient(to bottom, rgba(0,0,0,0.6) 0%, transparent 100%); padding: 6px; padding-bottom: 20px; pointer-events: none;">
                                    <div style="color: white; font-size: 11px; text-shadow: 0 1px 2px rgba(0,0,0,0.8); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" :title="img.name">{{ img.name }}</div>
                                </div>
                                <div v-if="selectedImages.includes(img.name)" style="position: absolute; top: 6px; right: 6px; background: #82318E; color: white; width: 22px; height: 22px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">✓</div>
                                
                                <div style="position: absolute; bottom: 6px; left: 6px; right: 6px; display: flex; justify-content: space-between; align-items: flex-end; pointer-events: none;">
                                    <div v-if="img.split !== 'unassigned'" style="background: rgba(43,108,176,0.9); color: white; border-radius: 4px; padding: 2px 5px; font-size: 10px; font-weight: bold; box-shadow: 0 1px 2px rgba(0,0,0,0.2);">
                                        {{ img.split.toUpperCase() }}
                                    </div>
                                    <div v-else></div>
                                    <div v-if="img.category" style="background: rgba(56,161,105,0.9); color: white; border-radius: 4px; padding: 2px 5px; font-size: 10px; font-weight: bold; max-width: 60px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; box-shadow: 0 1px 2px rgba(0,0,0,0.2);" :title="img.category">
                                        {{ img.category }}
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- PAGINATION -->
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 16px; padding-top: 12px; border-top: 1px solid #e2e8f0;">
                            <div style="font-size: 14px; color: #718096;">
                                共 <strong>{{ totalImages }}</strong> 项检索结果 (已选 <strong style="color: #82318E;">{{ selectedImages.length }}</strong> 项)
                            </div>
                            <div style="display: flex; gap: 8px; align-items: center;">
                                <button :disabled="currentPage <= 1" @click="changePage(currentPage - 1)" style="padding: 6px 12px; border: 1px solid #e2e8f0; background: white; border-radius: 4px; cursor: pointer;" :style="currentPage <= 1 ? 'opacity: 0.5; cursor: not-allowed;' : ''">上一页</button>
                                <span style="font-size: 14px; color: #4a5568;">第 {{ currentPage }} / {{ Math.max(1, Math.ceil(totalImages / pageSize)) }} 页</span>
                                <button :disabled="currentPage >= Math.ceil(totalImages / pageSize)" @click="changePage(currentPage + 1)" style="padding: 6px 12px; border: 1px solid #e2e8f0; background: white; border-radius: 4px; cursor: pointer;" :style="currentPage >= Math.ceil(totalImages / pageSize) ? 'opacity: 0.5; cursor: not-allowed;' : ''">下一页</button>
                            </div>
                        </div>
                    </div>

                    <!-- BATCH OPERATIONS SIDEBAR -->
                    <div style="flex: 1; min-width: 250px; max-width: 320px; border-left: 1px solid #e2e8f0; padding-left: 20px; display: flex; flex-direction: column; overflow-y: auto;">
                        <h4 style="margin: 0 0 16px 0; color: #2d3748; font-weight: bold; border-bottom: 2px solid #82318E; padding-bottom: 8px; display: inline-block;">快速批量操作</h4>
                        
                        <!-- Batch Sub-section: Category -->
                        <div style="background: #f7fafc; padding: 16px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #e2e8f0;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                                <span style="font-size: 14px; font-weight: bold; color: #4a5568;">🏷️ 赋予标签类别</span>
                                <button @click="showCategoryManager = true" style="background: none; border: none; color: #3182ce; cursor: pointer; font-size: 12px; text-decoration: underline;">类别管理</button>
                            </div>
                            <div style="position: relative;">
                                <input type="text" v-model="batchTag" @focus="showCategoryDropdown = true" @blur="hideCategoryDropdown" placeholder="🔍 搜索或输入类别..." style="width: 100%; box-sizing: border-box; border: 1px solid #cbd5e0; padding: 8px; border-radius: 6px; margin-bottom: 12px; outline: none; background: white; font-size: 13px;">
                                <div v-if="showCategoryDropdown && filteredCategories.length > 0" style="position: absolute; top: 100%; left: 0; right: 0; background: white; border: 1px solid #cbd5e0; border-radius: 6px; max-height: 200px; overflow-y: auto; z-index: 20; margin-top: -10px; margin-bottom: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                                    <div v-for="cat in filteredCategories" :key="cat" @mousedown.prevent="batchTag = cat; showCategoryDropdown = false" style="padding: 8px 12px; cursor: pointer; transition: background 0.2s; font-size: 13px;" onmouseover="this.style.backgroundColor='#edf2f7'" onmouseout="this.style.backgroundColor='transparent'">
                                        {{ cat }}
                                    </div>
                                </div>
                            </div>
                            <button @click="applyBatchTag" style="width: 100%; padding: 10px; background: #82318E; color: white; border: none; border-radius: 6px; font-weight: bold; cursor: pointer; transition: opacity 0.2s;" :style="selectedImages.length === 0 || !batchTag ? 'opacity: 0.5; cursor: not-allowed;' : ''">
                                给选中项打标
                            </button>
                        </div>

                        <!-- Batch Sub-section: Split -->
                        <div style="background: #f7fafc; padding: 16px; border-radius: 8px; border: 1px solid #e2e8f0;">
                            <div style="margin-bottom: 12px;">
                                <span style="font-size: 14px; font-weight: bold; color: #4a5568;">🗂️ 分配数据分集属性</span>
                            </div>
                            <select v-model="batchSplit" style="width: 100%; border: 1px solid #cbd5e0; padding: 8px; border-radius: 6px; margin-bottom: 12px; outline: none; background: white;">
                                <option value="train">训练集 (Train)</option>
                                <option value="val">验证集 (Val)</option>
                                <option value="test">测试集 (Test)</option>
                            </select>
                            <button @click="applyBatchSplit" style="width: 100%; padding: 10px; background: #3182ce; color: white; border: none; border-radius: 6px; font-weight: bold; cursor: pointer; transition: opacity 0.2s;" :style="selectedImages.length === 0 ? 'opacity: 0.5; cursor: not-allowed;' : ''">
                                更改所属分集
                            </button>
                        </div>
                        
                        
                        <div style="margin-top: auto; padding-top: 20px;">
                            <button @click="selectAllVisible" style="width: 100%; padding: 8px; background: white; color: #4a5568; border: 1px solid #cbd5e0; border-radius: 6px; margin-bottom: 8px; cursor: pointer;">全选当前页可见图像</button>
                            <button @click="selectedImages = []" style="width: 100%; padding: 8px; background: white; color: #e53e3e; border: 1px solid #feb2b2; border-radius: 6px; cursor: pointer;">清空全部选择状态</button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Detail Tab: Detection -->
            <div v-else-if="detailTab === 'detection'" style="flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; background: white; border-radius: 12px; border: 1px dashed #cbd5e0;">
                <div style="font-size: 48px; margin-bottom: 16px;">🎯</div>
                <h3 style="font-weight: bold; color: #4a5568; margin-bottom: 8px;">通用检测 / 分割打标基站</h3>
                <p style="color: #a0aec0; text-align: center; max-width: 400px; margin-bottom: 24px;">此处将全量嵌入现有的 SAM 高维像素级分割接口与 YOLO Bounding Box 拉框模块，统一导出标准 txt 坐标系。</p>
                <button style="padding: 10px 24px; background: #edf2f7; color: #4a5568; font-weight: bold; border-radius: 8px; border: none; cursor: pointer;">正在研发载入协议...</button>
            </div>
            
        </div>

        <!-- Global Toast Notification -->
        <div v-if="toastMessage" :style="{ position: 'fixed', top: '24px', left: '50%', transform: 'translateX(-50%)', padding: '12px 24px', borderRadius: '8px', background: toastType === 'error' ? '#e53e3e' : '#38a169', color: 'white', fontWeight: 'bold', boxShadow: '0 10px 15px rgba(0,0,0,0.2)', zIndex: 99999, transition: 'all 0.3s ease', display: 'flex', alignItems: 'center', gap: '8px' }">
            <span v-if="toastType === 'success'">✅</span>
            <span v-if="toastType === 'error'">❌</span>
            {{ toastMessage }}
        </div>

        <!-- Modal: Delete Selected Images Confirmation -->
        <div v-if="showDeleteImagesModal" style="position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.6); display: flex; align-items: center; justify-content: center; z-index: 9999;">
            <div style="background: white; padding: 32px; border-radius: 12px; width: 400px; box-shadow: 0 10px 25px rgba(0,0,0,0.3);">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
                    <span style="font-size: 32px;">⚠️</span>
                    <h3 style="margin: 0; color: #e53e3e;">永久删除选中样本？</h3>
                </div>
                <p style="color: #4a5568; line-height: 1.6; margin-bottom: 24px;">
                    确实要永久摧毁选中的 <strong style="color: #e53e3e;">{{ selectedImages.length }}</strong> 张样本数据吗？<br><br>此操作将直接从文件系统抹除原始数据及配套锚框，<strong>不可逆转！</strong>
                </p>
                <div style="display: flex; justify-content: flex-end; gap: 12px;">
                    <button @click="showDeleteImagesModal = false" style="padding: 10px 20px; background: white; color: #4a5568; border: 1px solid #cbd5e0; border-radius: 6px; cursor: pointer; font-weight: bold;">取消</button>
                    <button @click="confirmDeleteSelectedImages" style="padding: 10px 20px; background: #e53e3e; color: white; border: none; border-radius: 6px; cursor: pointer; font-weight: bold;">确认彻底删除</button>
                </div>
            </div>
        </div>

        <!-- Modal: Delete Database Confirmation -->
        <div v-if="databaseToDelete" style="position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; z-index: 1000;">
            <div style="background: white; padding: 32px; border-radius: 12px; width: 400px; box-shadow: 0 10px 25px rgba(0,0,0,0.2);">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
                    <span style="font-size: 32px;">⚠️</span>
                    <h3 style="margin: 0; color: #e53e3e;">彻底删除数据仓库？</h3>
                </div>
                
                <p style="color: #4a5568; line-height: 1.6; margin-bottom: 24px;">
                    您确定要彻底删除数据仓库 <strong>[{{ databaseToDelete }}]</strong> 吗？<br><br>此操作将永久抹除该资产下所有的图像文件及标注文件，<strong>且操作不可逆！</strong>
                </p>
                
                <div style="display: flex; justify-content: flex-end; gap: 12px;">
                    <button @click="databaseToDelete = null" style="padding: 10px 20px; background: white; color: #4a5568; border: 1px solid #cbd5e0; border-radius: 6px; cursor: pointer; font-weight: bold;">取消</button>
                    <button @click="deleteDatabase" style="padding: 10px 20px; background: #e53e3e; color: white; border: none; border-radius: 6px; cursor: pointer; font-weight: bold;">确认彻底删除</button>
                </div>
            </div>
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

        <!-- Modal: Create Version -->
        <div v-if="showCreateVersionModal" style="position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; z-index: 1000;">
            <div style="background: white; padding: 32px; border-radius: 12px; width: 400px; box-shadow: 0 10px 25px rgba(0,0,0,0.2);">
                <h3 style="margin-top: 0; margin-bottom: 20px;">新建标注库版本</h3>
                <div style="margin-bottom: 24px;">
                    <label style="display: block; font-size: 14px; font-weight: bold; margin-bottom: 8px;">版本名称 (如 v2)</label>
                    <input type="text" v-model="newVersionName" placeholder="例如: v2" style="width: 100%; padding: 10px; border: 1px solid #e2e8f0; border-radius: 6px; box-sizing: border-box;" />
                </div>
                <div style="display: flex; justify-content: flex-end; gap: 12px;">
                    <button @click="showCreateVersionModal = false" style="padding: 10px 20px; background: white; border: 1px solid #cbd5e0; border-radius: 6px; cursor: pointer;">取消</button>
                    <button @click="createVersion" style="padding: 10px 20px; background: #82318E; color: white; border: none; border-radius: 6px; cursor: pointer;">创建版本</button>
                </div>
            </div>
        </div>

        <!-- Modal: Category Manager -->
        <div v-if="showCategoryManager" style="position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; z-index: 1000;">
            <div style="background: white; padding: 32px; border-radius: 12px; width: 500px; box-shadow: 0 10px 25px rgba(0,0,0,0.2); display: flex; flex-direction: column; max-height: 80vh;">
                <h3 style="margin-top: 0; margin-bottom: 20px;">全局类别配置 (Categories)</h3>
                
                <div style="display: flex; gap: 8px; margin-bottom: 24px;">
                    <input type="text" v-model="newCategoryName" @keyup.enter="addCategory" placeholder="输入新的分类标签名称..." style="flex: 1; padding: 10px; border: 1px solid #e2e8f0; border-radius: 6px; box-sizing: border-box; outline: none;" />
                    <button @click="addCategory" style="padding: 10px 20px; background: #38a169; color: white; border: none; border-radius: 6px; cursor: pointer; font-weight: bold;">+ 新增</button>
                </div>
                
                <div style="flex: 1; overflow-y: auto; border: 1px solid #edf2f7; border-radius: 8px; padding: 8px;">
                    <div v-if="categories.length === 0" style="text-align: center; color: #a0aec0; padding: 20px;">尚未创建任何类别节点</div>
                    <div v-for="cat in categories" :key="cat" style="display: flex; justify-content: space-between; align-items: center; padding: 12px; border-bottom: 1px solid #edf2f7;">
                        <span style="font-weight: bold; color: #4a5568;">{{ cat }}</span>
                        <div style="display: flex; gap: 8px;">
                            <button @click="deleteCategory(cat)" style="background: #fed7d7; color: #c53030; border: none; border-radius: 4px; padding: 4px 8px; cursor: pointer; font-size: 12px;">删除</button>
                        </div>
                    </div>
                </div>
                
                <div style="display: flex; justify-content: flex-end; margin-top: 24px;">
                    <button @click="showCategoryManager = false" style="padding: 10px 24px; background: #82318E; color: white; border: none; border-radius: 6px; cursor: pointer; font-weight: bold;">完成管理</button>
                </div>
            </div>
        </div>
        
        <!-- Modal: Image Preview -->
        <div v-if="previewImageUrl" @click="previewImageUrl = null" style="position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.85); display: flex; align-items: center; justify-content: center; z-index: 2000; cursor: zoom-out;">
            <img :src="previewImageUrl" style="max-width: 90%; max-height: 90%; object-fit: contain; border-radius: 8px; box-shadow: 0 10px 25px rgba(0,0,0,0.5);" @click.stop />
            <button @click="previewImageUrl = null" style="position: absolute; top: 20px; right: 20px; background: white; color: black; border: none; border-radius: 50%; width: 40px; height: 40px; font-size: 20px; cursor: pointer; display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 10px rgba(0,0,0,0.2);">×</button>
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
        // Navigation State
        const currentView = ref('list');
        const detailTab = ref('content');
        const selectedDataset = ref(null);

        // Core App State
        const databases = ref([]);
        const isLoading = ref(false);
        const showCreateModal = ref(false);
        const newDbName = ref('');
        const newDbDesc = ref('');
        const databaseToDelete = ref(null);

        // Annotation View State
        const versions = ref([]);
        const currentVersion = ref('');
        const categories = ref([]);
        const filterSplit = ref('all');
        const filterStatus = ref('all');
        const filterCategory = ref('all');
        const searchQuery = ref('');
        const currentPage = ref(1);
        const pageSize = ref(40);
        const totalImages = ref(0);
        const paginatedImages = ref([]);
        const isImagesLoading = ref(false);

        // Batch Selection State
        const selectedImages = ref([]);
        const batchTag = ref('');
        const batchSplit = ref('train');
        const showCategoryDropdown = ref(false);

        const filteredCategories = computed(() => {
            if (!batchTag.value) {
                return categories.value;
            }
            const query = batchTag.value.toLowerCase();
            return categories.value.filter(cat => cat.toLowerCase().includes(query));
        });

        const hideCategoryDropdown = () => {
            setTimeout(() => {
                showCategoryDropdown.value = false;
            }, 100);
        };

        // Custom Toast System
        const toastMessage = ref('');
        const toastType = ref('success');
        let toastTimeout = null;

        const showToast = (msg, type = 'success') => {
            toastMessage.value = msg;
            toastType.value = type;
            if (toastTimeout) clearTimeout(toastTimeout);
            toastTimeout = setTimeout(() => { toastMessage.value = ''; }, 3000);
        };

        // Modals
        const showCreateVersionModal = ref(false);
        const newVersionName = ref('');
        const showCategoryManager = ref(false);
        const newCategoryName = ref('');
        const showDeleteImagesModal = ref(false);

        /* Database API Calls */
        const fetchDatabases = async () => {
            isLoading.value = true;
            try {
                const res = await window.axios.get(`${API_BASE}/dataset/list`);
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
            if (!newDbName.value.trim()) return showToast("请输入数据仓库名称！", "error");
            try {
                const res = await window.axios.post(`${API_BASE}/dataset/create`, {
                    project_name: newDbName.value.trim(),
                    description: newDbDesc.value.trim()
                });
                if (res.data.status === 'success') {
                    showCreateModal.value = false;
                    newDbName.value = '';
                    newDbDesc.value = '';
                    fetchDatabases();
                    showToast("工程仓建立成功！", "success");
                }
            } catch (err) {
                showToast(`创建失败: ${err.response?.data?.detail || err.message}`, "error");
            }
        };

        const deleteDatabase = async () => {
            if (!databaseToDelete.value) return;
            try {
                const res = await window.axios.delete(`${API_BASE}/dataset/${databaseToDelete.value}`);
                if (res.data.status === 'success') {
                    databaseToDelete.value = null;
                    fetchDatabases();
                    showToast("工程仓已彻底摧毁", "success");
                }
            } catch (err) {
                showToast(`删除失败: ${err.response?.data?.detail || err.message}`, "error");
            }
        };

        /* Detailed View Entry */
        const openDataset = (db) => {
            selectedDataset.value = db;
            currentView.value = 'detail';
            detailTab.value = 'content';
            resetAnnotationState();
            fetchCategories();
            fetchVersions();
        };

        const closeDataset = () => {
            selectedDataset.value = null;
            currentView.value = 'list';
        };

        /* Category APIs */
        const fetchCategories = async () => {
            if (!selectedDataset.value) return;
            try {
                const res = await window.axios.get(`${API_BASE}/dataset/${selectedDataset.value.name}/categories`);
                if (res.data.status === 'success') categories.value = res.data.data;
            } catch (err) {
                console.error("Failed to fetch categories:", err);
            }
        };

        const addCategory = async () => {
            if (!newCategoryName.value.trim()) return;
            try {
                const res = await window.axios.post(`${API_BASE}/dataset/${selectedDataset.value.name}/categories`, {
                    name: newCategoryName.value.trim()
                });
                if (res.data.status === 'success') {
                    categories.value = res.data.data;
                    newCategoryName.value = '';
                }
            } catch (err) {
                console.error("Failed to add category:", err);
            }
        };

        const deleteCategory = async (catName) => {
            try {
                const res = await window.axios.delete(`${API_BASE}/dataset/${selectedDataset.value.name}/categories/${catName}`);
                if (res.data.status === 'success') categories.value = res.data.data;
            } catch (err) {
                console.error("Failed to delete category:", err);
            }
        };

        const fetchVersions = async () => {
            if (!selectedDataset.value) return;
            try {
                const res = await window.axios.get(`${API_BASE}/dataset/${selectedDataset.value.name}/versions`);
                if (res.data.status === 'success') {
                    versions.value = res.data.data;
                    if (versions.value.length > 0 && !currentVersion.value) {
                        currentVersion.value = versions.value[0].version;
                    }
                    fetchImages(); // Always fetch images, even if versions.value is empty
                }
            } catch (err) {
                console.error("Failed to fetch versions:", err);
            }
        };

        const createVersion = async () => {
            if (!newVersionName.value.trim()) return;
            try {
                const res = await window.axios.post(`${API_BASE}/dataset/${selectedDataset.value.name}/versions`, {
                    version: newVersionName.value.trim(),
                    description: "Created via frontend"
                });
                if (res.data.status === 'success') {
                    showCreateVersionModal.value = false;
                    newVersionName.value = '';
                    fetchVersions();
                }
            } catch (err) {
                showToast(`创建版本失败: ${err.response?.data?.detail || err.message}`, "error");
            }
        };

        // NEW LOGIC FOR STATS, PREVIEW, AND UPLOAD/DELETE
        const previewImageUrl = ref(null);
        const datasetStats = ref({ total: 0, splits: { train: 0, val: 0, test: 0, unassigned: 0 }, annotated: 0, unannotated: 0 });

        const getImageUrl = (imageName) => {
            return `${API_BASE}/dataset/${selectedDataset.value.name}/image/${imageName}`;
        };

        const openPreviewImage = (imageName) => {
            previewImageUrl.value = getImageUrl(imageName);
        };

        const fetchDatasetStats = async () => {
            if (!selectedDataset.value) return;
            const targetVersion = currentVersion.value || "v1";
            try {
                const res = await window.axios.get(`${API_BASE}/dataset/${selectedDataset.value.name}/stats?version=${targetVersion}`);
                if (res.data && res.data.status === 'success') {
                    datasetStats.value = res.data.data;
                }
            } catch (err) {
                console.error("Failed to fetch dataset stats (Non-fatal):", err);
            }
        };

        const handleFileUpload = async (event) => {
            const files = event.target.files;
            if (!files || files.length === 0) return;

            const formData = new FormData();
            for (let i = 0; i < files.length; i++) {
                formData.append('files', files[i]);
            }

            try {
                isImagesLoading.value = true;
                const res = await window.axios.post(`${API_BASE}/dataset/${selectedDataset.value.name}/images`, formData, {
                    headers: { 'Content-Type': 'multipart/form-data' }
                });

                if (res && res.data && res.data.status === 'success') {
                    event.target.value = '';
                    showToast(`成功上传 ${res.data.files.length} 张样本`);
                    resetPaginationAndFetch();
                }
            } catch (err) {
                console.error("Upload error details:", err);
                event.target.value = '';
                const errMsg = err.response && err.response.data && err.response.data.detail ? err.response.data.detail : err.message;
                showToast(`上传失败: ${errMsg}`, "error");
            } finally {
                isImagesLoading.value = false;
            }
        };

        const deleteSelectedImages = () => {
            if (selectedImages.value.length === 0) return;
            showDeleteImagesModal.value = true;
        };

        const confirmDeleteSelectedImages = async () => {
            try {
                isImagesLoading.value = true;
                const res = await window.axios.delete(`${API_BASE}/dataset/${selectedDataset.value.name}/images`, {
                    data: { images: selectedImages.value }
                });
                if (res.data.status === 'success') {
                    selectedImages.value = [];
                    showDeleteImagesModal.value = false;
                    showToast("样本已成功摧毁", "success");
                    resetPaginationAndFetch();
                }
            } catch (err) {
                showToast(`删除失败: ${err.response?.data?.detail || err.message}`, "error");
            } finally {
                isImagesLoading.value = false;
            }
        };

        /* Image Reading APIs */
        const fetchImages = async () => {
            if (!selectedDataset.value) return;
            const targetVersion = currentVersion.value || "v1";
            isImagesLoading.value = true;
            try {
                // Remove trailing wildcard behaviors in URL search strings
                const qSearch = encodeURIComponent(searchQuery.value);
                const url = `${API_BASE}/dataset/${selectedDataset.value.name}/images?version=${targetVersion}&split=${filterSplit.value}&annotated=${filterStatus.value}&category=${encodeURIComponent(filterCategory.value)}&search=${qSearch}&page=${currentPage.value}&page_size=${pageSize.value}`;
                const res = await window.axios.get(url);
                if (res.data.status === 'success') {
                    paginatedImages.value = res.data.data;
                    totalImages.value = res.data.total;
                }
            } catch (err) {
                console.error("Failed to fetch images:", err);
            } finally {
                isImagesLoading.value = false;
                fetchDatasetStats(); // Refresh stats whenever we fetch images
            }
        };

        const resetPaginationAndFetch = () => {
            currentPage.value = 1;
            selectedImages.value = []; // Clear selection when filter changes
            fetchImages();
        };

        const changePage = (p) => {
            currentPage.value = p;
            selectedImages.value = [];
            fetchImages();
        };

        /* Selection & Batch Operations */
        const toggleSelectImage = (imgName) => {
            const idx = selectedImages.value.indexOf(imgName);
            if (idx === -1) selectedImages.value.push(imgName);
            else selectedImages.value.splice(idx, 1);
        };

        const selectAllVisible = () => {
            paginatedImages.value.forEach(img => {
                if (!selectedImages.value.includes(img.name)) {
                    selectedImages.value.push(img.name);
                }
            });
        };

        const applyBatchTag = async () => {
            if (selectedImages.value.length === 0 || !batchTag.value) return;
            try {
                const res = await window.axios.post(`${API_BASE}/dataset/${selectedDataset.value.name}/annotations/${currentVersion.value}/batch`, {
                    images: selectedImages.value,
                    tag: batchTag.value
                });
                if (res.data.status === 'success') {
                    selectedImages.value = [];
                    fetchImages(); // Refresh current page
                }
            } catch (err) {
                showToast(`批量打标失败: ${err.message}`, "error");
            }
        };

        const applyBatchSplit = async () => {
            if (selectedImages.value.length === 0 || !batchSplit.value) return;
            try {
                const res = await window.axios.post(`${API_BASE}/dataset/${selectedDataset.value.name}/images/split`, {
                    images: selectedImages.value,
                    split: batchSplit.value
                });
                if (res.data.status === 'success') {
                    selectedImages.value = [];
                    showToast("批量切分分配成功", "success");
                    fetchImages(); // Refresh current page
                }
            } catch (err) {
                showToast(`批量分配失败: ${err.message}`, "error");
            }
        };

        const resetAnnotationState = () => {
            filterSplit.value = 'all';
            filterStatus.value = 'all';
            filterCategory.value = 'all';
            searchQuery.value = '';
            currentPage.value = 1;
            selectedImages.value = [];
            versions.value = [];
            currentVersion.value = '';
            batchTag.value = '';
            batchSplit.value = 'train';
            datasetStats.value = { total: 0, splits: { train: 0, val: 0, test: 0, unassigned: 0 }, annotated: 0, unannotated: 0 };
        };

        // Tab watcher
        watch(detailTab, (newVal) => {
            if (newVal === 'classification' || newVal === 'content') {
                filterSplit.value = 'all';
                filterStatus.value = 'all';
                filterCategory.value = 'all';
                searchQuery.value = '';
                resetPaginationAndFetch();
            }
            if (newVal === 'content') {
                fetchDatasetStats();
            }
        });

        // Initialize App
        onMounted(() => {
            fetchDatabases();
        });

        return {
            currentView, detailTab, selectedDataset, openDataset, closeDataset,
            databases, isLoading, showCreateModal, newDbName, newDbDesc, databaseToDelete,
            fetchDatabases, createDatabase, deleteDatabase,

            versions, currentVersion, categories,
            filterSplit, filterStatus, filterCategory, searchQuery,
            currentPage, pageSize, totalImages, paginatedImages, isImagesLoading,
            datasetStats, previewImageUrl, getImageUrl, openPreviewImage,
            fetchDatasetStats, handleFileUpload, deleteSelectedImages,

            selectedImages, batchTag, batchSplit,
            showCreateVersionModal, newVersionName,
            showCategoryManager, newCategoryName,
            showCategoryDropdown, filteredCategories, hideCategoryDropdown,

            toastMessage, toastType, showDeleteImagesModal, confirmDeleteSelectedImages,

            resetPaginationAndFetch, changePage,
            toggleSelectImage, selectAllVisible,
            fetchCategories, addCategory, deleteCategory,
            createVersion, applyBatchTag, applyBatchSplit
        };
    }
};
