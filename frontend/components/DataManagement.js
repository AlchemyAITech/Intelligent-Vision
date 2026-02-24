import { ref } from 'vue';

export default {
    name: 'DataManagement',
    template: `
    <div style="padding: 24px;">
        <h2 style="font-size: 24px; font-weight: bold; margin-bottom: 24px; color: #82318E;">数据管理仓</h2>
        
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
    `,
    setup() {
        const uploadFiles = () => {
            alert('功能即将开放：批量上传 / 本地挂载目录');
        };

        return {
            uploadFiles
        };
    }
};
