import { ref, onMounted, computed, nextTick, watch } from 'vue';
import ImageSource from './ImageSource.js';

export default {
    name: 'FaceLab',
    components: {
        ImageSource
    },
    template: `
    <div class="face-lab">
        <!-- Modal Overlay System -->
        <div v-if="isFaceModalOpen || isGestureModalOpen" class="modal-overlay" style="position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.4); backdrop-filter:blur(10px); z-index:9999; display:flex; align-items:center; justify-content:center; padding: 20px;">
            <!-- Face Capture Modal (Standard Size Refined) -->
            <div v-if="isFaceModalOpen" class="modal-content" style="background:white; width:90%; max-width:65vh; border-radius:24px; padding:25px; box-shadow:0 30px 60px rgba(0,0,0,0.3); position:relative; border:1px solid #eee; display:flex; flex-direction:column; gap:15px;">
                <h3 style="margin:0; color:#333; display:flex; align-items:center; gap:8px;">
                    <span style="font-size:24px;">📸</span> 实时人像人员录入
                </h3>
                
                <!-- Name Input Inside Modal -->
                <div class="modal-input-group">
                    <label style="font-size:13px; color:#666; font-weight:700; margin-bottom:5px; display:block;">人员姓名 (必填):</label>
                    <input type="text" v-model="enrollName" placeholder="请输入姓名，例如: 张三" 
                           style="width:100%; padding:12px; border:2px solid #667eea; border-radius:10px; font-size:16px; outline:none; transition:border-color 0.3s;"
                           :style="{ borderColor: !enrollName.trim() ? '#ff4d4f' : '#667eea' }">
                    <p v-if="!enrollName.trim()" style="color:#ff4d4f; font-size:11px; margin-top:4px;">⚠️ 必须填写姓名后才能执行存盘操作</p>
                </div>
                
                <div style="background:#000; border-radius:16px; overflow:hidden; aspect-ratio:4/3; display:flex; align-items:center; justify-content:center; position:relative; border:2px solid #eee;">
                    <!-- Raw stream as background to ensure no black screen during handshake -->
                    <ImageSource 
                        @stream-frame="onManageStreamFrame" 
                        :hideVideoPreview="false" 
                        :hideCaptureBtn="true"
                        :minimal="true"
                        defaultSourceType="webcam" 
                        :autoStart="true" 
                        style="width:100%; height:100%; object-fit: contain;" 
                    />
                    
                    <!-- Detection overlay (the red box stream) -->
                    <img v-if="managePreviewUrl" :src="managePreviewUrl" 
                         style="position:absolute; top:0; left:0; width:100%; height:100%; object-fit:contain; z-index:5;">
                    
                    <div v-if="showSuccessCheck" style="position:absolute; top:0; left:0; width:100%; height:100%; background:rgba(76, 175, 80, 0.4); display:flex; align-items:center; justify-content:center; z-index:100; pointer-events:none;">
                        <div style="background:white; padding:15px 30px; border-radius:50px; font-size:16px; color:#2e7d32; font-weight:700; box-shadow:0 10px 20px rgba(0,0,0,0.2);">✅ 已成功录入</div>
                    </div>
                </div>

                <div style="display:flex; gap:12px;">
                    <button class="btn-primary" 
                            style="flex:2; padding:14px; font-size:16px; font-weight:700; border-radius:12px; transition:opacity 0.3s;" 
                            @click="onEnrollAction(null)" 
                            :disabled="!enrollName.trim() || !managePreviewUrl"
                            :style="{ opacity: enrollName.trim() ? 1 : 0.5 }">
                        确定并保存
                    </button>
                    <button class="btn-secondary" style="flex:1; padding:14px; border-radius:12px;" @click="closeModals">取消</button>
                </div>
            </div>

            <!-- Gesture Capture Modal (Standard Size Refined) -->
            <div v-if="isGestureModalOpen" class="modal-content" style="background:white; width:90%; max-width:65vh; border-radius:24px; padding:25px; box-shadow:0 30px 60px rgba(0,0,0,0.3); position:relative; border:1px solid #eee;">
                <h3 style="margin-top:0; color:#333;">✋ 手势拓扑录制</h3>
                <div style="background:#000; border-radius:16px; overflow:hidden; aspect-ratio:16/9; display:flex; align-items:center; justify-content:center; position:relative; border:2px solid #eee;">
                    <ImageSource 
                        @stream-frame="onGestureStreamFrame" 
                        :hideVideoPreview="true" 
                        :minimal="true"
                        defaultSourceType="webcam" 
                        :autoStart="true" 
                        style="position:absolute; opacity:0; pointer-events:none;" 
                    />
                    <div v-if="!gesturePreviewUrl" style="color:#555; text-align:center;">
                         <h4>正在启动手部探测网络...</h4>
                    </div>
                    <img v-else :src="gesturePreviewUrl" style="width:100%; height:100%; object-fit:contain;">
                    
                    <div v-if="showSuccessCheck" style="position:absolute; top:0; left:0; width:100%; height:100%; background:rgba(76, 175, 80, 0.4); display:flex; align-items:center; justify-content:center; z-index:100; pointer-events:none;">
                        <div style="background:white; padding:15px 30px; border-radius:50px; font-size:18px; color:#2e7d32; font-weight:700; box-shadow:0 10px 20px rgba(0,0,0,0.2);">✅ 手势模型已存盘</div>
                    </div>
                </div>

                <div style="margin-top:15px; min-height:85px;">
                     <!-- Permanent Status Indicator -->
                     <div v-if="captureLandmarks" style="background:#f1f8e9; padding:10px; border-radius:10px; border:1px solid #c5e1a5; margin-bottom:10px; color:#33691e; font-size:14px; font-weight:700; text-align:center;">
                        🟢 信号稳定：检测到手部特征
                     </div>
                     <div v-else style="background:#fff1f0; padding:10px; border-radius:10px; border:1px solid #ffa39e; margin-bottom:10px; color:#cf1322; font-size:14px; font-weight:700; text-align:center;">
                        🔴 信号丢失：请将手部置于镜头中心
                     </div>
                     <input type="text" v-model="newGestureName" placeholder="命名此手势 (如: OK)" style="width:100%; padding:12px; font-size:18px; border:1px solid #ddd; border-radius:10px;">
                </div>

                <div style="margin-top:20px; display:flex; gap:12px;">
                    <button class="btn-primary" style="flex:2; padding:15px; font-size:18px; font-weight:800; border-radius:12px;" @click="saveGesture" :disabled="!captureLandmarks || !newGestureName">录存到自定义库</button>
                    <button class="btn-secondary" style="flex:1; padding:15px; border-radius:12px;" @click="closeModals">取消</button>
                </div>
            </div>
        </div>

        <!-- NEW: Sample Transfer Modal -->
        <div v-if="isTransferModalOpen" class="modal-overlay" style="position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.4); backdrop-filter:blur(10px); z-index:99999; display:flex; align-items:center; justify-content:center; padding: 20px;">
            <div class="modal-content" style="background:white; width:90%; max-width:450px; border-radius:24px; padding:25px; box-shadow:0 30px 60px rgba(0,0,0,0.3); border:1px solid #eee;">
                <h3 style="margin-top:0; color:#333;">📂 转移图片样本</h3>
                <p style="font-size:13px; color:#666;">转移图片: <b>{{ transferSource.filename }}</b></p>
                
                <div style="margin-top:20px; display:flex; flex-direction:column; gap:15px;">
                    <div class="modal-input-group">
                        <label style="font-size:12px; font-weight:700; color:#999; margin-bottom:5px; display:block;">1. 选择目标分类分组</label>
                        <select v-model="transferTargetGroup" style="width:100%; padding:10px; border-radius:8px; border:1px solid #ddd;">
                            <option value="known">👥 已知人员 (Known)</option>
                            <option value="strangers">❓ 陌生人库 (Strangers)</option>
                            <option value="blacklist">🚫 黑名单隔离 (Blacklist)</option>
                        </select>
                    </div>

                    <div class="modal-input-group">
                        <label style="font-size:12px; font-weight:700; color:#999; margin-bottom:5px; display:block;">2. 选择/输入目标姓名</label>
                        <div style="display:flex; gap:8px;">
                            <select v-model="transferTargetPersonName" style="flex:1; padding:10px; border-radius:8px; border:1px solid #ddd;">
                                <option value="">-- 请选择现有人员 --</option>
                                <option v-for="p in groupedBankList" :key="p.name" :value="p.name">{{ p.name }}</option>
                            </select>
                        </div>
                        <input type="text" v-model="transferTargetManualName" placeholder="或在此输入新姓名..." style="width:100%; margin-top:8px; padding:10px; border-radius:8px; border:1px solid #ddd;">
                    </div>
                </div>

                <div style="margin-top:25px; display:flex; gap:12px;">
                    <button class="btn-primary" style="flex:2; padding:12px; border-radius:10px;" @click="confirmTransfer" :disabled="!finalTransferName">确认转移</button>
                    <button class="btn-secondary" style="flex:1; padding:12px; border-radius:10px;" @click="isTransferModalOpen = false">取消</button>
                </div>
            </div>
        </div>

        <div class="lab-header" style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
            <h2>👤 人脸与手势实验室</h2>
            <div class="tabs" style="margin-bottom:0; border-bottom:none;">
                <button :class="{active: activeSubTab === 'analyze'}" @click="closeDetails('analyze')">智能分析</button>
                <button :class="{active: activeSubTab === 'manage'}" @click="closeDetails('manage')">人员库维护</button>
                <button :class="{active: activeSubTab === 'gesture'}" @click="closeDetails('gesture')">手势库管理</button>
            </div>
        </div>

        <div class="layout-grid">
            <div class="sidebar-panel">
                <div v-if="activeSubTab === 'analyze'">
                    <h3>1. 图像源</h3>
                    <ImageSource @image-selected="onImageSelected" @stream-frame="onStreamFrame" />
                    <hr>
                    
                    <h3>2. 分析选项</h3>
                    <div class="control-group checkbox">
                        <label><input type="checkbox" v-model="doRecognition" @change="requestAnalysis"> 人脸识别 (Recognition)</label>
                    </div>
                    <div class="control-group checkbox">
                        <label><input type="checkbox" v-model="doLandmarks" @change="requestAnalysis"> 面部网格 (Landmarks)</label>
                    </div>
                    <div class="control-group checkbox">
                        <label><input type="checkbox" v-model="doHands" @change="requestAnalysis"> 手势分析 (Hands & Gestures)</label>
                    </div>
                    
                    <hr>
                    <h3>3. 智能调节 (阈值)</h3>
                    <div class="control-group">
                        <label style="display:flex; justify-content:space-between;">
                            <span>人脸识别敏感度:</span>
                            <span style="color:var(--primary-accent); font-weight:700;">{{ recThreshold }}</span>
                        </label>
                        <input type="range" v-model.number="recThreshold" min="0" max="1" step="0.05" @change="requestAnalysis" style="width:100%;">
                        <p style="font-size:11px; color:#666; margin-top:4px;">💡 低值越严格，高值越宽松。默认 0.6。</p>
                    </div>
                    <div class="control-group" style="margin-top:15px;">
                        <label style="display:flex; justify-content:space-between;">
                            <span>手势匹配精细度:</span>
                            <span style="color:var(--primary-accent); font-weight:700;">{{ gestThreshold }}</span>
                        </label>
                        <input type="range" v-model.number="gestThreshold" min="0.05" max="0.5" step="0.01" @change="requestAnalysis" style="width:100%;">
                        <p style="font-size:11px; color:#666; margin-top:4px;">💡 设定判定自定义手势的距离容差。默认 0.15。</p>
                    </div>
                </div>

                <div v-else-if="activeSubTab === 'manage' && !selectedPerson">
                    <h3>1. 库概况</h3>
                    <div class="stat-card" style="background:rgba(162, 28, 175, 0.05); padding:15px; border-radius:10px; border:1px solid var(--panel-border);">
                        <div style="font-size:12px; color:var(--text-muted);">已知人员</div>
                        <div style="font-size:24px; font-weight:700; color:var(--primary-accent);">{{ knownCount }}</div>
                    </div>
                    <div class="stat-card" style="background:rgba(255, 165, 0, 0.05); padding:15px; border-radius:10px; border:1px solid rgba(255,165,0,0.2); margin-top:10px;">
                        <div style="font-size:12px; color:var(--text-muted);">陌生人 (待分类)</div>
                        <div style="font-size:24px; font-weight:700; color: orange;">{{ strangerCount }}</div>
                    </div>
                    <div class="stat-card" style="background:rgba(255, 0, 0, 0.05); padding:15px; border-radius:10px; border:1px solid rgba(255,0,0,0.2); margin-top:10px;">
                        <div style="font-size:12px; color:var(--text-muted);">黑名单防范名录</div>
                        <div style="font-size:24px; font-weight:700; color: red;">{{ blacklists.length }}</div>
                    </div>
                    
                    <hr>
                    <div style="background:var(--panel-bg); padding:15px; border-radius:10px; border:1px solid var(--panel-border);">
                         <p style="font-size:12px; color:#666; margin-bottom:10px;">点击下方按钮开启人脸录入专用通道。</p>
                         <button class="btn-primary" style="width:100%;" @click="isFaceModalOpen = true">� 录入新成员</button>
                    </div>
                </div>
                
                <div v-else-if="activeSubTab === 'manage' && selectedPerson">
                    <h3>操作区</h3>
                    <div style="background:var(--panel-bg); padding:15px; border-radius:10px; border:1px solid var(--panel-border);">
                        <p style="font-size:13px; color:var(--text-muted); margin-top:0;">为 <b>{{ selectedPerson.name }}</b> 传入新样本：</p>
                        <button class="btn-primary" style="width:100%;" @click="startEnrollToSelected">📸 开发采集弹窗</button>
                        <hr style="margin: 15px 0;">
                        <button class="btn-secondary" style="width:100%; margin-bottom:10px;" @click="onRename(selectedPerson.name)">更改该成员名</button>
                        <button v-if="!selectedPerson.is_stranger" class="btn-secondary" :style="{ color: selectedPerson.is_blacklist ? 'green' : 'red', borderColor: selectedPerson.is_blacklist ? 'green' : 'red' }" style="width:100%;" @click="toggleBlacklist(selectedPerson.name)">
                            {{ selectedPerson.is_blacklist ? '从黑名单中解除' : '移入黑名单防范' }}
                        </button>
                    </div>
                </div>

                <div v-else>
                    <h3>1. 自建手势录制</h3>
                    <p style="font-size:12px; color:var(--text-muted);">点击下方按钮开启全屏手势采集器。</p>
                    <button class="btn-primary" style="width:100%;" @click="isGestureModalOpen = true">✋ 开启手势拓扑录制</button>
                </div>
            </div>

            <div class="main-panel">
                <div v-if="activeSubTab === 'analyze'">
                    <h3>3. 多维分析视图</h3>
                    <div v-if="!imageUrl" class="empty-state">请加载一张图片。</div>
                    <div v-else>
                        <div v-if="isLoading" class="loading-state">正在处理中...</div>
                        <div v-else>
                            <div class="result-display" style="background:#000; border-radius:12px; overflow:hidden; display:flex; align-items:center; justify-content:center; height:55vh; min-height:55vh; width:100%;">
                                <img :src="resultImageUrl || imageUrl" class="preview-img" style="height:100%; width:100%; object-fit: contain;">
                            </div>
                            <!-- Proportional Stats -->
                            <div class="stats-panel" style="margin-top:20px; display:flex; gap:20px;">
                                <div class="stat-card" style="background:rgba(162, 28, 175, 0.05); padding:15px; border-radius:10px; border:1px solid var(--panel-border); flex:1;">
                                    <div style="font-size:12px; color:var(--text-muted);">检测到人脸</div>
                                    <div style="font-size:24px; font-weight:700; color:var(--primary-accent);">{{ facesCount }}</div>
                                </div>
                                <div class="stat-card" v-if="isThreat" style="background:rgba(255, 0, 0, 0.1); padding:15px; border-radius:10px; border:1px solid rgba(255,0,0,0.3); flex:1.5;">
                                    <div style="font-size:12px; color:red;">安全状态 (Security)</div>
                                    <div style="font-size:20px; font-weight:700; color:red;">⚠️ 危险警告: 发现黑名单人物</div>
                                </div>
                                <div class="stat-card" v-else style="background:rgba(0, 255, 0, 0.05); padding:15px; border-radius:10px; border:1px solid rgba(0,255,0,0.2); flex:1;">
                                    <div style="font-size:12px; color:green;">安全状态</div>
                                    <div style="font-size:20px; font-weight:700; color:green;">✅ 环境安全</div>
                                </div>
                            </div>
                            <!-- Capture Button -->
                            <div v-if="detectedFaces.length > 0" style="margin-top:15px; background:rgba(255,255,255,0.1); padding:15px; border-radius:12px; border:1px solid var(--panel-border);">
                                <div style="font-weight:600; margin-bottom:10px;">发现 {{ detectedFaces.length }} 个目标</div>
                                <div style="display:flex; gap:10px; flex-wrap:wrap;">
                                    <button class="btn-primary" style="padding:6px 15px; font-size:13px;" @click="captureAllStrangers">📸 抓拍所有陌生人</button>
                                    <button class="btn-secondary" style="padding:6px 15px; font-size:13px;" @click="fetchBank">🔄 刷新人员库</button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div v-else-if="activeSubTab === 'manage'">
                    <div v-if="!selectedPerson">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
                            <h3 style="margin:0;">人员库名单分组</h3>
                            <button class="btn-secondary" @click="fetchBank">🔄 刷新数据</button>
                        </div>
                        <div v-if="bankList.length === 0" class="empty-state">库中暂无人物数据。</div>
                        
                        <div v-if="knownPersons.length > 0" style="margin-bottom: 30px;">
                            <h4 style="color:var(--primary-accent); border-bottom:1px solid var(--panel-border); padding-bottom:5px; margin-top:0;">👥 已知人员 ({{ knownPersons.length }})</h4>
                            <div class="face-bank-grid" style="display:grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap:15px;">
                                <div v-for="person in knownPersons" :key="person.name" class="person-card" style="background:var(--panel-bg); border:1px solid var(--panel-border); border-left:6px solid var(--primary-accent); border-radius:12px; padding:15px; cursor:pointer;" @click="openPersonDetails(person)">
                                    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                                        <div>
                                            <div style="font-weight:700; font-size:17px;">{{ person.name }}</div>
                                            <div style="font-size:12px; color:var(--text-muted); margin-top:2px;">样本数量: {{ person.count }}</div>
                                        </div>
                                        <div style="display:flex; gap:5px;" @click.stop>
                                            <button class="btn-secondary" style="padding:4px 8px; font-size:11px; color:red; border-color:rgba(255,0,0,0.2);" @click="deletePerson(person.name)">删除人员</button>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div v-if="strangers.length > 0" style="margin-bottom: 30px;">
                            <h4 style="color:orange; border-bottom:1px solid rgba(255,165,0,0.3); padding-bottom:5px;">❓ 陌生人库 ({{ strangers.length }})</h4>
                            <div class="face-bank-grid" style="display:grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap:15px;">
                                <div v-for="person in strangers" :key="person.name" class="person-card" style="background:var(--panel-bg); border:1px solid var(--panel-border); border-left:6px solid #faad14; border-radius:12px; padding:15px; cursor:pointer;" @click="openPersonDetails(person)">
                                    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                                        <div>
                                            <div style="font-weight:700; font-size:17px; display:flex; align-items:center; gap:8px;">
                                                {{ person.name }}
                                                <span style="font-size:10px; background:#faad14; color:white; padding:2px 6px; border-radius:4px;">STRANGER</span>
                                            </div>
                                            <div style="font-size:12px; color:var(--text-muted); margin-top:2px;">样本数量: {{ person.count }}</div>
                                        </div>
                                        <div style="display:flex; gap:5px;" @click.stop>
                                            <button class="btn-secondary" style="padding:4px 8px; font-size:11px;" @click="onPromote(person.name)">转正登记</button>
                                            <button class="btn-secondary" style="padding:4px 8px; font-size:11px; color:red; border-color:rgba(255,0,0,0.2);" @click="deletePerson(person.name)">移除</button>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div v-if="blacklists.length > 0" style="margin-bottom: 30px;">
                            <h4 style="color:red; border-bottom:1px solid rgba(255,0,0,0.3); padding-bottom:5px;">🚫 黑名单隔离 ({{ blacklists.length }})</h4>
                            <div class="face-bank-grid" style="display:grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap:15px;">
                                <div v-for="person in blacklists" :key="person.name" class="person-card" style="background:var(--panel-bg); border:1px solid var(--panel-border); border-left:6px solid #ff4d4f; border-radius:12px; padding:15px; cursor:pointer;" @click="openPersonDetails(person)">
                                    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                                        <div>
                                            <div style="font-weight:700; font-size:17px; display:flex; align-items:center; gap:8px;">
                                                {{ person.name }}
                                            </div>
                                            <div style="font-size:12px; color:var(--text-muted); margin-top:2px;">样本数量: {{ person.count }}</div>
                                        </div>
                                        <div style="display:flex; gap:5px;" @click.stop>
                                            <button class="btn-secondary" style="padding:4px 8px; font-size:11px; color:red; border-color:rgba(255,0,0,0.2);" @click="deletePerson(person.name)">永久删除</button>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div v-else>
                        <!-- Selected Person Detail View -->
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
                            <div style="display:flex; align-items:center; gap:10px;">
                                <button class="btn-secondary" @click="selectedPerson = null">⬅ 返回人员编队</button>
                                <h3 style="margin:0;">[<span :style="{color: selectedPerson.is_blacklist ? 'red' : (selectedPerson.is_stranger ? 'orange' : 'var(--primary-accent)')}">{{ selectedPerson.name }}</span>] 的样本图集</h3>
                            </div>
                            <button class="btn-secondary" @click="fetchPersonSamples(selectedPerson.name)">🔄 刷新图像</button>
                        </div>

                        <div v-if="personSamples.length === 0" class="empty-state">该人员暂无图片样本记录。</div>
                        <div v-else style="display:grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap:15px;">
                            <div v-for="sample in personSamples" :key="sample" style="background:var(--panel-bg); border-radius:8px; overflow:hidden; border:1px solid var(--panel-border); text-align:center;">
                                <img :src="'/face_bank/' + (selectedPerson.is_stranger ? 'Strangers/' : '') + selectedPerson.name + '/' + sample" style="width:100%; height:130px; object-fit:cover;">
                                <div style="display:flex; border-top:1px solid var(--panel-border);">
                                    <button style="flex:1; padding:8px 0; background:transparent; border:none; border-right:1px solid var(--panel-border); cursor:pointer; font-size:12px; color:var(--text-main);" @click="transferSample(selectedPerson.name, sample)">转移...</button>
                                    <button style="flex:1; padding:8px 0; background:transparent; border:none; cursor:pointer; font-size:12px; color:red;" @click="deleteSample(selectedPerson.name, sample)">删除</button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div v-else>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
                        <h3 style="margin:0;">手势与体态姿势库</h3>
                        <button class="btn-secondary" @click="fetchGestures">🔄 刷新</button>
                    </div>

                    <div style="margin-bottom:30px;">
                        <h4 style="border-bottom:1px solid var(--panel-border); padding-bottom:5px;">原生系统基础手势</h4>
                        <div style="display:grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap:15px;">
                            <div v-for="(alias, name) in (gestureList.aliases || {})" :key="name" class="gesture-card" style="background:var(--panel-bg); border:1px solid var(--panel-border); border-radius:12px; padding:15px;">
                                <div style="font-weight:700; font-size:15px; color:var(--text-main); margin-bottom:8px;">{{ name === 'None' ? 'None (无手势)' : name }}</div>
                                <div>
                                    <input type="text" :value="alias" @blur="e => saveGestureAlias(name, e.target.value)" placeholder="添加自定义备注名..." style="width:100%; font-size:12px; padding:6px; border:1px solid rgba(0,0,0,0.1); border-radius:4px; box-sizing:border-box;">
                                </div>
                            </div>
                        </div>
                    </div>

                    <div>
                        <h4 style="border-bottom:1px solid var(--panel-border); padding-bottom:5px;">📸 自定义影像录制手势</h4>
                        <div v-if="!gestureList.custom || Object.keys(gestureList.custom).length === 0" class="empty-state">暂无自定义手势。使用左侧录像。</div>
                        <div v-else class="gesture-grid" style="display:grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap:15px;">
                            <div v-for="(landmarks, name) in gestureList.custom" :key="name" class="gesture-card" style="background:var(--panel-bg); border:1px solid var(--panel-border); border-radius:12px; padding:15px; position:relative;">
                                <div style="font-weight:700; font-size:18px; color:var(--primary-accent);">{{ name }}</div>
                                <div style="font-size:11px; color:var(--text-muted); margin-top:5px;">基于 21 点手部拓扑空间投影</div>
                                <button class="btn-secondary" @click="deleteGesture(name)" style="margin-top:10px; color:red; border-color:rgba(255,0,0,0.2); width:100%;">删除手势</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    `,
    setup() {
        // State
        const activeSubTab = ref('analyze');
        const imageUrl = ref('');
        const selectedFile = ref(null);
        const resultImageUrl = ref('');
        const isLoading = ref(false);
        const facesCount = ref(0);
        const detectedFaces = ref([]);
        const isThreat = ref(false);
        const doRecognition = ref(true);
        const doLandmarks = ref(true);
        const doHands = ref(true);
        const recThreshold = ref(0.6);
        const gestThreshold = ref(0.15);
        const lastRawAnalyzeB64 = ref('');

        const bankList = ref([]);
        const knownCount = ref(0);
        const strangerCount = ref(0);
        const knownPersons = computed(() => bankList.value.filter(p => !p.is_stranger && !p.is_blacklist));
        const strangers = computed(() => bankList.value.filter(p => p.is_stranger));
        const blacklists = computed(() => bankList.value.filter(p => p.is_blacklist));

        const enrollName = ref('');
        const selectedPerson = ref(null);
        const personSamples = ref([]);

        const gestureList = ref({ custom: {}, aliases: {} });
        const captureLandmarks = ref(null);
        const newGestureName = ref('');

        const isFaceModalOpen = ref(false);
        const isGestureModalOpen = ref(false);
        const showSuccessCheck = ref(false);

        // Transfer States
        const isTransferModalOpen = ref(false);
        const transferSource = ref({ name: '', filename: '' });
        const transferTargetGroup = ref('known');
        const transferTargetPersonName = ref('');
        const transferTargetManualName = ref('');

        const finalTransferName = computed(() => transferTargetManualName.value.trim() || transferTargetPersonName.value);

        const groupedBankList = computed(() => {
            if (transferTargetGroup.value === 'known') return knownPersons.value;
            if (transferTargetGroup.value === 'strangers') return strangers.value;
            if (transferTargetGroup.value === 'blacklist') return blacklists.value;
            return [];
        });

        const closeModals = () => {
            isFaceModalOpen.value = false;
            isGestureModalOpen.value = false;
            isTransferModalOpen.value = false;
            managePreviewUrl.value = '';
            gesturePreviewUrl.value = '';
            captureLandmarks.value = null;
            showSuccessCheck.value = false;
        };

        // Tab Switching
        const closeDetails = (tab) => {
            activeSubTab.value = tab;
            selectedPerson.value = null;
        };

        // Data Fetching
        const fetchBank = async () => {
            try {
                const res = await axios.get('/api/face/bank/list');
                bankList.value = res.data;
                knownCount.value = res.data.filter(p => !p.is_stranger && !p.is_blacklist).length;
                strangerCount.value = res.data.filter(p => p.is_stranger).length;

                if (selectedPerson.value) {
                    const found = res.data.find(p => p.name === selectedPerson.value.name);
                    if (found) {
                        selectedPerson.value = found;
                        fetchPersonSamples(found.name);
                    } else {
                        selectedPerson.value = null;
                    }
                }
            } catch (e) { }
        };

        const fetchGestures = async () => {
            try {
                const res = await axios.get('/api/face/bank/gestures');
                gestureList.value = {
                    custom: res.data.custom || {},
                    aliases: res.data.aliases || {}
                };
            } catch (e) { }
        };

        onMounted(() => {
            fetchBank();
            fetchGestures();
        });

        // Event Handlers for UI Callbacks
        const saveGestureAlias = async (name, alias) => {
            const formData = new FormData();
            formData.append('name', name);
            formData.append('alias', alias);
            await axios.post('/api/face/bank/save_gesture_alias', formData);
            fetchGestures();
        };

        const onImageSelected = (payload) => {
            if (payload && payload.data) {
                selectedFile.value = payload.data;
                imageUrl.value = URL.createObjectURL(payload.data);
                requestAnalysis();
            }
        };

        const startEnrollToSelected = () => {
            if (selectedPerson.value) {
                enrollName.value = selectedPerson.value.name;
                isFaceModalOpen.value = true;
            }
        };

        const onEnrollAction = async (payload) => {
            if (!enrollName.value.trim()) return;

            // In modal mode, if payload is null, it means "capture current preview"
            let fileToUpload = payload?.data;

            if (!fileToUpload && lastRawManageB64.value) {
                // Use the raw snapshot instead of the plotted managePreviewUrl
                const res = await fetch(lastRawManageB64.value);
                fileToUpload = await res.blob();
            }

            if (fileToUpload) {
                const formData = new FormData();
                let targetDirName = enrollName.value.trim();

                // If we are appending to a selected person (who might be a stranger)
                if (selectedPerson.value && selectedPerson.value.name === targetDirName) {
                    if (selectedPerson.value.is_stranger) {
                        targetDirName = "Strangers/" + targetDirName;
                    }
                }

                formData.append('name', targetDirName);
                formData.append('file', new File([fileToUpload], 'capture.jpg', { type: 'image/jpeg' }));

                try {
                    await axios.post('/api/face/bank/upload_sample', formData);
                    showSuccessCheck.value = true;
                    setTimeout(() => { showSuccessCheck.value = false; }, 1500);

                    if (!selectedPerson.value) {
                        // If we are in "New Person" mode, maybe the user wants to take more pics of the same new person? 
                        // Or if we are done, they can close. Let's keep it open but show success.
                    } else {
                        fetchPersonSamples(selectedPerson.value.name);
                    }
                    fetchBank();
                } catch (e) {
                    alert("❌ 录入失败：" + (e.response?.data?.detail || "后台检测未通过"));
                }
            }
        };

        const deletePerson = async (name) => {
            if (!confirm(`确定要永久删除人员 [${name}] 及其所有样本记录吗？此操作不可逆！`)) return;
            const formData = new FormData();
            formData.append('name', name);
            try {
                await axios.post('/api/face/bank/delete_person', formData);
                if (selectedPerson.value && selectedPerson.value.name === name) {
                    selectedPerson.value = null; // Close detail view if deleting the currently viewed person
                }
                fetchBank();
            } catch (e) {
                alert("删除失败");
            }
        };

        const captureAllStrangers = async () => {
            const hasData = selectedFile.value || lastRawAnalyzeB64.value;
            if (!hasData || detectedFaces.value.length === 0) {
                alert("未检测到可抓拍的人脸，请先确保画面中已正常出现检测选框。");
                return;
            }

            isLoading.value = true;
            let capturedCount = 0;
            try {
                const facesToCapture = JSON.parse(JSON.stringify(detectedFaces.value));
                for (const face of facesToCapture) {
                    const formData = new FormData();
                    if (selectedFile.value) {
                        formData.append('file', selectedFile.value);
                    } else {
                        const res = await fetch(lastRawAnalyzeB64.value);
                        const blob = await res.blob();
                        formData.append('file', new File([blob], 'stream_capture.jpg', { type: 'image/jpeg' }));
                    }
                    formData.append('bbox_json', JSON.stringify(face.bbox));
                    await axios.post('/api/face/bank/capture_stranger', formData);
                    capturedCount++;
                }
                alert(`抓拍完成，已成功将 ${capturedCount} 位人物存入库。`);
                fetchBank();
            } catch (e) {
                alert("抓拍过程中出错，请检查后台日志。");
            } finally {
                isLoading.value = false;
            }
        };

        const openPersonDetails = async (person) => {
            selectedPerson.value = person;
            fetchPersonSamples(person.name);
        };

        const fetchPersonSamples = async (name) => {
            const prefix = selectedPerson.value.is_stranger ? "Strangers/" : "";
            const res = await axios.get('/api/face/bank/samples/' + prefix + name);
            personSamples.value = res.data;
        };

        const deleteSample = async (name, filename) => {
            if (!confirm("确定要删除这张样本吗？一旦删除不可恢复。")) return;
            const formData = new FormData();
            formData.append('name', selectedPerson.value.is_stranger ? "Strangers/" + name : name);
            formData.append('filename', filename);
            await axios.post('/api/face/bank/delete_sample', formData);
            fetchBank();
        };

        const transferSample = (name, filename) => {
            transferSource.value = { name, filename };
            transferTargetPersonName.value = '';
            transferTargetManualName.value = '';
            isTransferModalOpen.value = true;
        };

        const confirmTransfer = async () => {
            const target = finalTransferName.value;
            if (!target) return;

            const formData = new FormData();
            formData.append('name', selectedPerson.value.is_stranger ? "Strangers/" + transferSource.value.name : transferSource.value.name);
            formData.append('filename', transferSource.value.filename);

            // Construct target path based on group
            let fullTarget = target;
            if (transferTargetGroup.value === 'strangers' && !target.startsWith('Strangers/')) {
                fullTarget = "Strangers/" + target;
            } else if (transferTargetGroup.value === 'blacklist' && !target.startsWith('Blacklist/')) {
                // Backend might not support Blacklist/ prefix in transfer directly if it's just a flag, 
                // but usually transfer to blacklist means moving to known then flagging.
                // However, based on existing code, Strangers/ is the only logical prefix.
            }

            formData.append('new_name', fullTarget);
            try {
                await axios.post('/api/face/bank/transfer_sample', formData);
                isTransferModalOpen.value = false;
                fetchBank();
            } catch (e) {
                alert("转移失败");
            }
        };

        const uploadSample = async (event, name) => {
            const file = event.target.files[0];
            if (!file) return;
            const formData = new FormData();
            formData.append('name', selectedPerson.value.is_stranger ? "Strangers/" + name : name);
            formData.append('file', file);
            try {
                await axios.post('/api/face/bank/upload_sample', formData);
                fetchBank();
            } catch (e) {
                alert("上传失败：" + (e.response?.data?.detail || "无法提取脸部特征或数据无效"));
            }
            event.target.value = '';
        };

        const onGestureImageSelected = (payload) => {
            // To capture landmarks, we use the analyze endpoint with hands enabled
            if (payload && payload.data) {
                const formData = new FormData();
                formData.append('file', payload.data);
                formData.append('do_hands', true);
                formData.append('do_recognition', false);
                formData.append('do_landmarks', false);
                formData.append('gest_threshold', gestThreshold.value);

                axios.post('/api/face/analyze_metadata', formData).then(res => {
                    if (res.data.hand_landmarks) {
                        captureLandmarks.value = res.data.hand_landmarks;
                    } else {
                        alert("🔴 提取失败：未能在此图像或镜头画面中识别到清晰的手部特征点架构。\n请换用更明显的手势或调整光线角度再试一次。");
                    }
                }).catch(err => {
                    alert("获取手势信息时出错，请确保服务连接正常。");
                });
            }
        };

        const saveGesture = async () => {
            if (!captureLandmarks.value || !newGestureName.value) return;
            const formData = new FormData();
            formData.append('name', newGestureName.value);
            formData.append('landmarks_json', JSON.stringify(captureLandmarks.value));
            try {
                await axios.post('/api/face/bank/save_gesture', formData);
                showSuccessCheck.value = true;
                setTimeout(() => { showSuccessCheck.value = false; }, 1500);

                // Clear name but keep stream going for potentially more gestures
                newGestureName.value = '';
                // No need to clear captureLandmarks yet, it will update with the stream
                fetchGestures();
            } catch (e) {
                alert("手势保存失败");
            }
        };

        const deleteGesture = async (name) => {
            const formData = new FormData();
            formData.append('name', name);
            await axios.post('/api/face/bank/delete_gesture', formData);
            fetchGestures();
        };

        const requestAnalysis = async () => {
            if (!selectedFile.value) return;
            isLoading.value = true;
            const formData = new FormData();
            formData.append('file', selectedFile.value);
            formData.append('do_recognition', doRecognition.value);
            formData.append('do_landmarks', doLandmarks.value);
            formData.append('do_hands', doHands.value);
            formData.append('rec_threshold', recThreshold.value);
            formData.append('gest_threshold', gestThreshold.value);

            try {
                const res = await axios.post('/api/face/analyze', formData);
                resultImageUrl.value = res.data.image_b64;
                facesCount.value = res.data.faces_count;
                detectedFaces.value = res.data.faces || [];
                isThreat.value = res.data.is_threat;
            } catch (err) {
                alert("分析失败");
            } finally {
                isLoading.value = false;
            }
        };

        const onRename = async (o) => {
            const n = prompt("将该成员重命名为:", o);
            if (n && n !== o) {
                const f = new FormData(); f.append('old_name', o); f.append('new_name', n);
                await axios.post('/api/face/bank/rename', f); fetchBank();
            }
        };

        const onPromote = async (o) => {
            const n = prompt("为改名陌生人赋予正名身份:", "");
            if (n) {
                const f = new FormData(); f.append('name', o); f.append('new_name', n);
                await axios.post('/api/face/bank/promote_stranger', f); fetchBank();
            }
        };

        const toggleBlacklist = async (n) => {
            const f = new FormData(); f.append('name', n);
            await axios.post('/api/face/bank/toggle_blacklist', f); fetchBank();
        };

        let isStreamProcessing = false;
        const onStreamFrame = async (b64) => {
            if (activeSubTab.value !== 'analyze' || isStreamProcessing) return;
            isStreamProcessing = true;
            lastRawAnalyzeB64.value = b64;

            if (!imageUrl.value) {
                imageUrl.value = b64;
            }

            try {
                const res = await fetch(b64);
                const blob = await res.blob();
                const file = new File([blob], 'stream.jpg', { type: 'image/jpeg' });

                const formData = new FormData();
                formData.append('file', file);
                formData.append('do_recognition', doRecognition.value);
                formData.append('do_landmarks', doLandmarks.value);
                formData.append('do_hands', doHands.value);
                formData.append('rec_threshold', recThreshold.value);
                formData.append('gest_threshold', gestThreshold.value);

                // Silent request to prevent UI loading flicker
                const response = await axios.post('/api/face/analyze', formData);
                resultImageUrl.value = response.data.image_b64;
                facesCount.value = response.data.faces_count;
                detectedFaces.value = response.data.faces || [];
                isThreat.value = response.data.is_threat;
            } catch (err) {
                // Ignore silent stream errors
            } finally {
                isStreamProcessing = false;
            }
        };

        const managePreviewUrl = ref('');
        const lastRawManageB64 = ref('');
        let isManageStreamProcessing = false;
        const onManageStreamFrame = async (b64) => {
            if (activeSubTab.value !== 'manage' || isManageStreamProcessing) return;
            isManageStreamProcessing = true;
            lastRawManageB64.value = b64;

            try {
                const res = await fetch(b64);
                const blob = await res.blob();
                const file = new File([blob], 'stream.jpg', { type: 'image/jpeg' });

                const formData = new FormData();
                formData.append('file', file);
                formData.append('do_hands', false);
                if (enrollName.value.trim()) {
                    formData.append('target_name', enrollName.value.trim());
                    formData.append('do_recognition', true); // Enable comparison with self
                } else {
                    formData.append('do_recognition', false);
                }

                const response = await axios.post('/api/face/analyze', formData);
                managePreviewUrl.value = response.data.image_b64;
            } catch (err) {
            } finally {
                isManageStreamProcessing = false;
            }
        };

        const gesturePreviewUrl = ref('');
        let isGestureStreamProcessing = false;
        const onGestureStreamFrame = async (b64) => {
            if (activeSubTab.value !== 'gesture' || isGestureStreamProcessing) return;
            isGestureStreamProcessing = true;

            try {
                const res = await fetch(b64);
                const blob = await res.blob();
                const file = new File([blob], 'stream.jpg', { type: 'image/jpeg' });

                const formData = new FormData();
                formData.append('file', file);
                formData.append('do_recognition', false);
                formData.append('do_landmarks', false); // No face tracking here
                formData.append('do_hands', true); // Request hand topology

                const response = await axios.post('/api/face/analyze', formData);
                gesturePreviewUrl.value = response.data.image_b64;

                // Sync landmarks for saving, even if not recognized as a specific gesture yet
                if (response.data.hands && response.data.hands.length > 0) {
                    captureLandmarks.value = response.data.hands[0].landmarks;
                } else {
                    captureLandmarks.value = null;
                }
            } catch (err) {
            } finally {
                isGestureStreamProcessing = false;
            }
        };

        // Clear previews when tab changes
        watch(activeSubTab, () => {
            managePreviewUrl.value = '';
            gesturePreviewUrl.value = '';
        });

        return {
            activeSubTab, closeDetails, imageUrl, resultImageUrl, isLoading, facesCount, detectedFaces, isThreat,
            doRecognition, doLandmarks, doHands, onImageSelected, onStreamFrame, requestAnalysis, captureAllStrangers,
            bankList, knownCount, strangerCount, knownPersons, strangers, blacklists,
            onRename, onPromote, toggleBlacklist, deletePerson, fetchBank,
            enrollName, onEnrollAction, startEnrollToSelected, selectedPerson, personSamples, openPersonDetails, fetchPersonSamples, deleteSample, transferSample, uploadSample,
            gestureList, fetchGestures, saveGestureAlias, onGestureImageSelected, captureLandmarks, newGestureName, saveGesture, deleteGesture,
            managePreviewUrl, onManageStreamFrame, gesturePreviewUrl, onGestureStreamFrame,
            isFaceModalOpen, isGestureModalOpen, closeModals, showSuccessCheck,
            recThreshold, gestThreshold, lastRawAnalyzeB64,
            isTransferModalOpen, transferSource, transferTargetGroup, transferTargetPersonName, transferTargetManualName, groupedBankList, confirmTransfer, finalTransferName
        };
    }
};
