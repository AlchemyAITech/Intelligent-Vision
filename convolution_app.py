
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import os
import sys

# --- Import from SRC ---
# Add current dir to path to ensure imports work if run from nested dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.image_ops import apply_kernel_2d, convolve_rgb, KERNELS
from src.utils import draw_red_box, get_center_matrix, load_source_image
from src.network import (
    SimpleCNN, tensor_to_img_array, load_mnist_data, 
    get_random_sample, calculate_accuracy
)
from streamlit_option_menu import option_menu

# PyTorch Imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# --- Config ---
st.set_page_config(
    page_title="智能视界 — 图像处理实验室",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS调整：减少顶部空白 ---
st.markdown("""
    <style>
        .block-container {
            padding-top: 0rem;
            padding-bottom: 0rem;
            margin-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

DATA_PATH = './data'

# --- 0. UI Helper ---
def render_dual_input(label, key_base, min_v, max_v, default_v, step=1, help_text=None):
    """
    Renders a slider and a number input that are synchronized.
    """
    state_key = f"{key_base}_val"
    if state_key not in st.session_state:
        st.session_state[state_key] = default_v
    
    def on_slider_change():
        st.session_state[state_key] = st.session_state[f"{key_base}_slide"]
    def on_num_change():
        st.session_state[state_key] = st.session_state[f"{key_base}_num"]

    c1, c2 = st.columns([3, 1])
    with c1:
        st.slider(
            label, min_v, max_v, 
            value=int(st.session_state[state_key]), 
            step=step, 
            key=f"{key_base}_slide", 
            on_change=on_slider_change,
            help=help_text
        )
    with c2:
        st.number_input(
            "数值", min_v, max_v, 
            value=int(st.session_state[state_key]), 
            step=step, 
            key=f"{key_base}_num", 
            on_change=on_num_change,
            label_visibility="hidden"
        )
    return st.session_state[state_key]





# --- 1. Color Space Lab ---
def render_color_space_lab(data_path):
    # st.header("🎨 图像的本质 (The Essence of Images)")
    
    # [v21 Layout Polish] 1:3 Ratio
    c_left, c_right = st.columns([1, 3])
    
    with c_left:
        st.markdown("### 1. 图像源")
        image1 = load_source_image("tab1")
        
        st.markdown("---")
        st.markdown("### 2. 分析设置")
        if image1:
            target_space = st.selectbox("目标色彩空间", ["Grayscale (L)", "YCbCr", "HSV", "RGB (Original)"], key="t1_space")
            
            converted_img = image1
            channels = []
            channel_names = []

            if target_space == "Grayscale (L)":
                converted_img = image1.convert('L')
                channels = [converted_img]
                channel_names = ['Gray']
            elif target_space == "YCbCr":
                converted_img = image1.convert('YCbCr')
                channels = converted_img.split()
                channel_names = ['Y (Luma)', 'Cb', 'Cr']
            elif target_space == "HSV":
                converted_img = image1.convert('HSV')
                channels = converted_img.split()
                channel_names = ['Hue', 'Saturation', 'Value']
            else:
                converted_img = image1
                channels = converted_img.split()
                channel_names = ['Red', 'Green', 'Blue']

    with c_right:
         st.markdown("### 3. 直观对比")
         if image1:
             ic1, ic2 = st.columns(2)
             with ic1:
                 st.image(image1, caption="原始图像 (RGB)", use_column_width=True)
             with ic2:
                 # Check if converted_img is defined (it should be if image1 is True)
                 # Handle display for different modes
                 if 'converted_img' in locals():
                     if converted_img.mode == 'HSV':
                          show_img = converted_img.convert('RGB')
                          caption_text = f"转换图像 ({target_space} -> RGB View)"
                     else:
                          show_img = converted_img
                          caption_text = f"转换图像 ({target_space})"
                     if show_img is not None:
                        # Fix TypeError: use_column_width instead for compatibility if needed
                        if converted_img.mode=='L':
                            st.image(show_img, caption=caption_text, use_column_width=True, clamp=True)
                        else:
                            st.image(show_img, caption=caption_text, use_column_width=True)
                 else:
                    st.error("Failed to convert image.")
             st.markdown("---")
             st.subheader(f"📊 通道拆分与矩阵数据 (Center 20x20)")
             
             if 'converted_img' in locals():
                 matrices = get_center_matrix(converted_img, size=20)
                 for idx, (ch_img, name) in enumerate(zip(channels, channel_names)):
                     st.markdown(f"**Channel {idx+1}: {name}**")
                     ch_img_with_box = draw_red_box(ch_img, size=20)
                     col_img, col_mat = st.columns([1, 2])
                     with col_img:
                         st.image(ch_img_with_box, caption=f"{name} (Red Box: Matrix Area)", use_column_width=True)
                     with col_mat:
                         if idx < len(matrices):
                             df = matrices[idx]
                             st.dataframe(df.style.background_gradient(cmap='gray', axis=None).format("{:.0f}"), height=300)
                             st.caption(f"Min: {df.min().min()}, Max: {df.max().max()}")
                     st.markdown("---")
         else:
             st.info("请加载一张图片。")

# --- 2. Convolution Lab ---
def render_convolution_lab(data_path):
    # st.header("⚙️ 卷积实验室")
    
    c_left, c_right = st.columns([1, 3])
    
    with c_left:
        st.markdown("### 1. 图像源")
        image2 = load_source_image("tab2")
        
        st.markdown("---")
        st.markdown("### 2. 卷积设置")
        if image2:
            process_mode = st.radio("处理模式 (输入)", ["RGB (彩色)", "Grayscale (灰度)"], key="t2_mode")
            st.markdown("---")
            category = st.selectbox("滤波器类别", list(KERNELS.keys()), key="t2_cat")
            kernel_name = st.selectbox("具体卷积核", list(KERNELS[category].keys()), key="t2_kname")
            kernel = KERNELS[category][kernel_name]
            st.markdown("#### 当前矩阵")
            st.code(str(np.round(kernel, 3)))
            st.markdown("---")
            invert_color = st.checkbox("反转颜色 (Invert Color)", key="t2_inv")

    with c_right:
        st.markdown("### 3. 处理结果")
        if image2:
             c_in, c_out = st.columns(2)
             if process_mode == "Grayscale (灰度)":
                 input_img = image2.convert('L')
                 is_gray_input = True
             else:
                 input_img = image2
                 is_gray_input = False
             
             with c_in:
                 st.subheader("输入图像")
                 if is_gray_input:
                      st.image(input_img, caption="Grayscale Input", use_column_width=True)
                 else:
                      st.image(input_img, caption="RGB Input", use_column_width=True)
             with c_out:
                 st.subheader("卷积结果")
                 with st.spinner("计算中..."):
                     if is_gray_input:
                         # Use helper from SRC
                         res_arr = apply_kernel_2d(np.array(input_img), kernel)
                         res_arr = np.clip(res_arr, 0, 255).astype(np.uint8)
                         result_img = Image.fromarray(res_arr)
                         if invert_color:
                             result_img = ImageOps.invert(result_img)
                         st.image(result_img, caption=f"Result: {kernel_name}", use_column_width=True)
                     else:
                         result_img = convolve_rgb(input_img, kernel)
                         if invert_color:
                             if result_img.mode == 'RGBA': result_img = result_img.convert('RGB')
                             result_img = ImageOps.invert(result_img)
                         st.image(result_img, caption=f"Result: {kernel_name}", use_column_width=True)
        else:
             st.info("请加载一张图片。")


# --- 3. Neural Network Lab (PyTorch) ---
def render_nn_lab():
    if not HAS_TORCH:
        st.error("未检测到 PyTorch! 请安装.")
    else:
        # st.header("🧠 卷积神经网络 (CNN) - 智能分类器")
        # st.markdown("构建并训练一个 CNN 模型来识别手写数字 MNIST。")
        
        # Session State Initialization
        if 'layer_configs' not in st.session_state: st.session_state.layer_configs = [16, 32]
        
        if 'model' not in st.session_state:
            st.session_state.model = SimpleCNN(layer_configs=st.session_state.layer_configs)
        if 'train_loss' not in st.session_state: st.session_state.train_loss = []
        
        # Accuracy State (Persistent)
        if 'train_acc' not in st.session_state: st.session_state.train_acc = 0.0
        if 'test_acc' not in st.session_state: st.session_state.test_acc = 0.0
        
        if 'last_viz_img' not in st.session_state: st.session_state.last_viz_img = None
        if 'last_prob_chart' not in st.session_state: st.session_state.last_prob_chart = None
        
        # Training Control States
        if 'is_training' not in st.session_state: st.session_state.is_training = False
        if 'stop_training' not in st.session_state: st.session_state.stop_training = False 
        if 'pause_training' not in st.session_state: st.session_state.pause_training = False 
        if 'current_epoch' not in st.session_state: st.session_state.current_epoch = 0

        tab_train, tab_test = st.tabs(["⚙️ 训练模块", "🧪 测试模块"])
        
        with tab_train:
            col_conf, col_viz = st.columns([1, 2])
            
            with col_viz:
                # [v20 Fix] Stable Layout & Accuracy Display
                c_head_1, c_head_2 = st.columns([1, 1])
                with c_head_1:
                    st.subheader("实时训练状态")
                with c_head_2:
                    if st.session_state.train_acc > 0:
                        st.metric("准确率 (Train / Test)", f"{st.session_state.train_acc:.1f}% / {st.session_state.test_acc:.1f}%")
                
                # Fixed Containers
                chart_container = st.container()
                
                st.markdown("### 动态特征图与预测")
                viz_container = st.container()
                
                with chart_container:
                     chart_placeholder = st.empty()
                     
                     status_text = st.empty()
                     # [User Request] Show pause message if paused
                     if st.session_state.pause_training:
                         msg = st.session_state.get('training_msg', 'Paused.')
                         status_text.info(f"{msg} (Paused)")
                         
                     if len(st.session_state.train_loss) > 0:
                         chart_placeholder.line_chart(st.session_state.train_loss)
                
                with viz_container:
                     c_feat, c_prob = st.columns([3, 2])
                     with c_feat:
                         st.markdown("**特征图可视化**")
                         feature_viz_placeholder = st.empty()
                     with c_prob:
                         st.markdown("**实时预测概率**")
                         prob_viz_placeholder = st.empty()

                     # Persistent Visuals (Always show if available)
                     if st.session_state.last_viz_img is not None:
                          feature_viz_placeholder.image(st.session_state.last_viz_img, caption="Last Snapshot", width=300)
                     if st.session_state.last_prob_chart is not None:
                          import altair as alt
                          # Handle if it is a dataframe
                          if isinstance(st.session_state.last_prob_chart, pd.DataFrame):
                              # [User Request] Show all 0-9 on Y axis
                              chart = alt.Chart(st.session_state.last_prob_chart).mark_bar().encode(
                                  x=alt.X('Probability', axis=alt.Axis(format='%')),
                                  y=alt.Y('Digit:O', title='Digit', scale=alt.Scale(domain=list(range(10)))),
                                  color=alt.value("#FF4B4B"),
                                  tooltip=['Digit', alt.Tooltip('Probability', format='.2%')]
                              ).properties(height=300)
                              prob_viz_placeholder.altair_chart(chart, use_container_width=True)
            
            with col_conf:
                st.subheader("训练配置")
                
                with st.expander("🛠️ 网络结构定制 (点击展开/收起)", expanded=False):
                    # 1. Select Number of Layers (Dual Input)
                    num_layers_val = render_dual_input(
                        "卷积层数量 (Num Layers)", "cfg_layers", 
                        1, 3, len(st.session_state.layer_configs)
                    )
                    
                    # 2. Configure Columns for Each Layer (Dual Input)
                    new_configs = []
                    for i in range(int(num_layers_val)):
                        default_val = st.session_state.layer_configs[i] if i < len(st.session_state.layer_configs) else (16 * (2**i))
                        
                        val = render_dual_input(
                            f"Layer {i+1} Channels", f"cfg_c{i}", 
                            4, 128, int(default_val), 4
                        )
                        new_configs.append(val)
                    
                    if st.button("保存架构并重置模型"):
                        st.session_state.layer_configs = new_configs
                        st.session_state.model = SimpleCNN(layer_configs=new_configs)
                        st.session_state.train_loss = []
                        st.session_state.train_acc = 0.0
                        st.session_state.test_acc = 0.0
                        st.session_state.current_epoch = 0
                        st.session_state.is_training = False
                        st.toast("模型架构已更新！", icon="✅")
                        st.rerun()

                st.write("---")
                lr = st.select_slider("学习率 (Learning Rate)", options=[0.001, 0.01, 0.1], value=0.01)
                num_epochs = st.slider("训练轮数 (Epochs)", 1, 10, 3) 
                batch_size = st.select_slider("批大小 (Batch Size)", options=[32, 64, 128], value=64)
                
                st.write("---")
                show_dynamic = st.checkbox("训练过程动态显示特征图 (所有层)", value=True)
                
                st.markdown("---")
                c_btn1, c_btn2 = st.columns(2)
                
                # [User Request] Refined Button Logic
                if not st.session_state.is_training:
                    # IDLE State
                    if c_btn1.button("🚀 开始训练", type="primary", use_container_width=True):
                         # [User Request] Always start from scratch
                         st.session_state.current_epoch = 0
                         st.session_state.train_loss = []
                         st.session_state.train_acc = 0.0
                         st.session_state.test_acc = 0.0
                         # Reset Model Weights
                         st.session_state.model = SimpleCNN(layer_configs=st.session_state.layer_configs)
                         
                         st.session_state.is_training = True
                         st.session_state.stop_training = False
                         st.session_state.pause_training = False 
                         st.rerun()
                
                elif st.session_state.is_training and not st.session_state.pause_training:
                    # RUNNING State
                    if c_btn1.button("⏹ 停止", type="secondary", use_container_width=True):
                        st.session_state.stop_training = True
                        st.session_state.is_training = False
                        st.session_state.pause_training = False
                        st.rerun()
                    if c_btn2.button("⏸ 暂停", use_container_width=True):
                        st.session_state.pause_training = True
                        st.rerun()
                
                elif st.session_state.is_training and st.session_state.pause_training:
                    # PAUSED State
                    if c_btn1.button("⏹ 停止", type="secondary", use_container_width=True):
                        st.session_state.stop_training = True
                        st.session_state.is_training = False
                        st.session_state.pause_training = False
                        st.rerun()
                    if c_btn2.button("▶️ 继续", type="primary", use_container_width=True):
                        st.session_state.pause_training = False
                        st.rerun()
            
            # --- Training Loop Execution ---
            if st.session_state.is_training and not st.session_state.pause_training:
                # Load Data
                with st.spinner("Loading Data..."):
                    train_data, test_data = load_mnist_data()
                    
                if train_data:
                    device = torch.device("cpu")
                    st.session_state.model.to(device)
                    optimizer = optim.Adam(st.session_state.model.parameters(), lr=lr)
                    criterion = nn.CrossEntropyLoss()
                    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
                    
                    start_epoch = st.session_state.current_epoch
                    
                    for epoch in range(start_epoch, num_epochs):
                        st.session_state.current_epoch = epoch 
                        running_loss = 0.0
                        viz_input_tensor, viz_label = get_random_sample(train_data)
                        
                        for i, (images, labels) in enumerate(train_loader):
                            if st.session_state.stop_training or st.session_state.pause_training:
                                break
                            
                            images, labels = images.to(device), labels.to(device)
                            optimizer.zero_grad()
                            outputs = st.session_state.model(images)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()
                            
                            running_loss += loss.item()
                            
                            if i % 10 == 0:
                                st.session_state.train_loss.append(loss.item())
                                chart_placeholder.line_chart(st.session_state.train_loss)
                                msg = f"Training... Epoch [{epoch+1}/{num_epochs}], Step [{i}]"
                                st.session_state.training_msg = msg
                                status_text.text(msg)
                            
                            # Dynamic Viz (Every 10 steps)
                            if show_dynamic and test_data and i % 10 == 0:
                                # [User Request] Switch image every 100 steps
                                if i % 100 == 0:
                                    viz_input_tensor, viz_label = get_random_sample(train_data)
                                    # Ensure tensor is on correct device if needed (CPU default here)
                                
                                # Feature Maps
                                feats = st.session_state.model.get_features(viz_input_tensor)
                                viz_names = []
                                for idx in range(1, len(st.session_state.layer_configs) + 1):
                                    if f'Conv{idx}' in feats: viz_names.append(f'Conv{idx}')
                                    if f'Pool{idx}' in feats: viz_names.append(f'Pool{idx}')
                                
                                layer_imgs = []
                                for layer_name in viz_names:
                                    f = feats[layer_name][0]
                                    # [User Request] Show 8 images per layer
                                    sub_imgs = [tensor_to_img_array(f[k]) for k in range(min(8, f.shape[0]))]
                                    def resize_pad(arr):
                                        im = Image.fromarray(arr).resize((40, 40), resample=Image.NEAREST)
                                        return np.array(im)
                                    row = np.hstack([resize_pad(img) for img in sub_imgs])
                                    layer_imgs.append(row)
                                
                                if layer_imgs:
                                    full = layer_imgs[0]
                                    for row in layer_imgs[1:]:
                                        if row.shape[1] < full.shape[1]:
                                             row = np.pad(row, ((0,0), (0, full.shape[1]-row.shape[1])), 'constant')
                                        elif row.shape[1] > full.shape[1]:
                                             full = np.pad(full, ((0,0), (0, row.shape[1]-full.shape[1])), 'constant')
                                        full = np.vstack([full, np.zeros((5, full.shape[1]), dtype=np.uint8), row])
                                    
                                    st.session_state.last_viz_img = full
                                    feature_viz_placeholder.image(full, caption=f"Layers (True: {viz_label})", width=300)
                                
                                # Prob Chart
                                with torch.no_grad():
                                    out_viz = st.session_state.model(viz_input_tensor)
                                    probs = F.softmax(out_viz, dim=1)[0].cpu().numpy()
                                
                                import altair as alt
                                df_probs = pd.DataFrame({"Digit": range(10), "Probability": probs})
                                st.session_state.last_prob_chart = df_probs
                                # [User Request] Show all 0-9 on Y axis
                                chart = alt.Chart(df_probs).mark_bar().encode(
                                    x=alt.X('Probability', axis=alt.Axis(format='%')),
                                    y=alt.Y('Digit:O', title='Digit', scale=alt.Scale(domain=list(range(10)))),
                                    color=alt.value("#FF4B4B"),
                                    tooltip=['Digit', alt.Tooltip('Probability', format='.2%')]
                                  ).properties(height=200)
                                prob_viz_placeholder.altair_chart(chart, use_container_width=True)
 
                        if st.session_state.stop_training or st.session_state.pause_training:
                            break
                    
                        if st.session_state.stop_training:
                            status_text.warning("Training Stopped.")
                            st.session_state.is_training = False
                            st.session_state.pause_training = False # Ensure pause is also reset
                            st.rerun() # Immediate rerun to show Start button
                        elif st.session_state.pause_training:
                            msg = st.session_state.get('training_msg', 'Paused.')
                            status_text.info(f"{msg} (Paused)")
                            st.rerun()
                        elif st.session_state.current_epoch >= num_epochs - 1:
                            # [User Request] Calculate Accuracy on Finish
                            with st.spinner("Calculating Accuracy..."):
                                t_loader = DataLoader(train_data, batch_size=1000, shuffle=False)
                                v_loader = DataLoader(test_data, batch_size=1000, shuffle=False)
                                
                                train_acc = calculate_accuracy(st.session_state.model, t_loader, device)
                                test_acc = calculate_accuracy(st.session_state.model, v_loader, device)
                                
                                st.session_state.train_acc = train_acc
                                st.session_state.test_acc = test_acc
                            
                            status_text.success(f"Finished! Train Acc: {train_acc:.1f}%, Test Acc: {test_acc:.1f}%")
                            st.session_state.is_training = False
                            st.session_state.pause_training = False
                            st.rerun()

        with tab_test:
            st.subheader("模型测试")
            t_col1, t_col2 = st.columns([1, 2])
            with t_col1:
                test_source = st.radio("来源", ["随机抽取", "上传"], key="test_src")
                if test_source == "随机抽取":
                    if st.button("抽取测试图"):
                         _, test_data = load_mnist_data()
                         if test_data:
                             input_tensor, label = get_random_sample(test_data)
                             inv_tensor = input_tensor[0] * 0.3081 + 0.1307
                             inv_tensor = torch.clamp(inv_tensor, 0, 1)
                             display_img = transforms.ToPILImage()(inv_tensor)
                             st.session_state['current_test_img'] = (input_tensor, display_img, label)
                else:
                    uploaded_test = st.file_uploader("上传 28x28 图片", type=["png", "jpg"])
                    if uploaded_test:
                        pil_img = Image.open(uploaded_test).convert('L').resize((28, 28))
                        display_img = pil_img
                        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                        input_tensor = transform(pil_img).unsqueeze(0)
                        st.session_state['current_test_img'] = (input_tensor, display_img, None)
                
                if 'current_test_img' in st.session_state:
                    input_tensor, display_img, label = st.session_state['current_test_img']
                    st.image(display_img, width=150)
                    if label is not None: st.text(f"True: {label}")
                    
                    if st.button("预测", key="test_pred_btn"):
                        st.session_state.model.eval()
                        with torch.no_grad():
                            output = st.session_state.model(input_tensor)
                            _, pred = torch.max(output, 1)
                            probs = F.softmax(output, dim=1)[0].numpy()
                            feats = st.session_state.model.get_features(input_tensor)
                            st.session_state['prediction_result'] = (pred.item(), probs, feats)
            
            with t_col2:
                if 'prediction_result' in st.session_state:
                    pred, probs, feats = st.session_state['prediction_result']
                    st.markdown(f"### 预测: **{pred}**")
                    
                    import altair as alt
                    df_probs = pd.DataFrame({"Digit": range(10), "Prob": probs})
                    chart = alt.Chart(df_probs).mark_bar().encode(
                        x=alt.X('Prob', axis=alt.Axis(format='%', title='概率')),
                        y=alt.Y('Digit:O', title='数字', scale=alt.Scale(domain=list(range(10)))),
                        color=alt.value("#FF4B4B"),
                        tooltip=['Digit', alt.Tooltip('Prob', format='.2%')]
                    ).properties(height=300)
                    st.altair_chart(chart, use_container_width=True)
                    
                    st.markdown("### 全层特征图")
                    with st.container(height=500, border=True):
                        # [v20 Fix] Ensure all keys are selectable
                        all_keys = list(feats.keys())
                        layers = st.multiselect("选择层", all_keys, default=all_keys)
                        for name in layers:
                            f = feats[name] 
                            C = f.shape[1]
                            st.write(f"**{name}** ({C} channels)")
                            cols = st.columns(8) # Show 8 columns
                            for i in range(min(C, 32)): 
                                with cols[i%8]:
                                    arr = tensor_to_img_array(f[0, i])
                                    st.image(arr, clamp=True, use_column_width=True)


# --- 4. YOLO Visual Tasks (Refactored) ---
def render_yolo_tab():
    try:
        from src.yolo_manager import YOLOManager
        import cv2
        import numpy as np # Ensure numpy is available
        
        # Initialize Manager
        manager = YOLOManager()
        
    except ImportError as e:
        st.error(f"依赖缺失或模块加载失败: {e}")
        return

    # st.header("👁️ YOLO 视觉任务 (Visual Tasks)")
    
    # Tabs for Task Type
    tab_names = ["物体检测 (Detect)", "图像分割 (Segment)", "姿态估计 (Pose)", "目标追踪 (Tracking)"]
    tabs = st.tabs(tab_names)
    
    # Standard Tasks (Detect, Segment, Pose)
    for i in range(3):
        with tabs[i]:
            render_yolo_task_content(tab_names[i], manager)
            
    # Tracking Tab
    with tabs[3]:
        render_yolo_tracking_content(manager)

def render_yolo_task_content(task_type, manager):
    import cv2
    import os
    
    # --- Layout: [1 (Config), 2 (Result)] ---
    c_conf, c_res = st.columns([1, 2])
    
    model = None
    input_source = None
    input_source_path = None # For video/local
    is_stream = False
    
    with c_conf:
        # --- 1. Input Source (Top) ---
        st.markdown("#### 1. 输入源 (Input)")
        source_type = st.radio("选择输入", ["🖼️ 图片上传 (Upload)", "📂 本地文件 (Local)", "📷 摄像头 (Webcam)"], 
                               key=f"src_{task_type}")
        
        # Input Widgets
        if source_type == "🖼️ 图片上传 (Upload)":
            uploaded_file = st.file_uploader("上传图片或视频", type=['jpg', 'png', 'jpeg', 'mp4', 'avi', 'mov'], key=f"up_{task_type}")
            if uploaded_file:
                # Save to local
                upload_dir = "images/uploads"
                if not os.path.exists(upload_dir): os.makedirs(upload_dir, exist_ok=True)
                file_path = os.path.join(upload_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                input_source_path = file_path
                
        elif source_type == "📂 本地文件 (Local)":
            local_dir = "images"
            if not os.path.exists(local_dir): os.makedirs(local_dir, exist_ok=True)
            valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.mp4', '.avi', '.mov')
            files = [f for f in os.listdir(local_dir) if f.lower().endswith(valid_exts)]
            if files:
                selected_file = st.selectbox("选择文件", files, key=f"loc_{task_type}")
                input_source_path = os.path.join(local_dir, selected_file)
            else:
                st.info(f"请将文件放入 `{local_dir}`")

        elif source_type == "📷 摄像头 (Webcam)":
            is_stream = True

        # --- 2. Parameter Settings ---
        st.markdown("#### 2. 参数设置 (Params)")
        
        with st.container(height=500):
            prompts = {}
            
            # Model Options based on Task
            if "Detect" in task_type:
                model_opts = ["yolo11n.pt", "yolo26n.pt", "yolo26s.pt", "yolo11s.pt", "yolov8n.pt"]
            elif "Segment" in task_type:
                model_opts = ["yolo11n-seg.pt", "yolo26n-seg.pt", "yolov8n-seg.pt"]
            elif "Pose" in task_type:
                model_opts = ["yolo11n-pose.pt", "yolo26n-pose.pt", "yolov8n-pose.pt"]
            else: # SAM
                # Added SAM 2.1 and SAM 3
                model_opts = ["sam2.1_b.pt", "sam2.1_t.pt", "mobile_sam.pt", "sam_b.pt"]
                
            model_name = st.selectbox("选择模型", model_opts, key=f"mdl_{task_type}")
            
            if "SAM" in task_type:
                 st.info("💡 SAM 3.0/2.1 支持文本和多点提示 (需最新依赖)")
                 sam_mode = st.radio("分割模式", ["全自动 (Auto)", "文本提示 (Text)", "框提示 (Box)"], key=f"sam_mode_{task_type}")
                 
                 if sam_mode == "文本提示 (Text)":
                     txt_prompt = st.text_input("输入提示词 (英文)", "person", key=f"sam_txt_{task_type}")
                     # Ultralytics SAM uses 'labels' or specific args? 
                     # Currently passing as kwargs to model() via manager
                     # Note: SAM 2 official API uses `text_prompts`? 
                     # Let's try passing 'prompt' or strict SAM API if documented.
                     # Ultralytics docs say for SAM 2: model.predict("asset.jpg", bboxes=[...])
                     # Text prompt support in Ultralytics wrapper is limited compared to Meta's repo.
                     # But let's assume standard k-v passing.
                     # Wait, SAM 3 supports open vocabulary with text.
                     # We will pass it as 'txt_prompts' or similar in manager.
                     if txt_prompt:
                        prompts['labels'] = [txt_prompt] # Warning: Might strictly need int labels for SAM 1/2
                        # For SAM 3 it might be prompts.
                        # We will rely on our manager to handle kwargs.
                        pass
                 elif sam_mode == "框提示 (Box)":
                     st.caption("模拟框坐标 [x1, y1, x2, y2]")
                     # Simplified slider for demo
                     b_x1 = st.slider("X1", 0, 640, 100, key=f"bx1_{task_type}")
                     b_y1 = st.slider("Y1", 0, 640, 100, key=f"by1_{task_type}")
                     b_x2 = st.slider("X2", 0, 640, 300, key=f"bx2_{task_type}")
                     b_y2 = st.slider("Y2", 0, 640, 300, key=f"by2_{task_type}")
                     prompts['bboxes'] = [[b_x1, b_y1, b_x2, b_y2]]
    
            # Load Model via Manager
            with st.spinner(f"正在加载/下载模型 {model_name}..."):
                model = manager.load_model(model_name)
            
            if not model:
                st.error("模型加载失败")
                return
                
            conf_thres = 0.25
            iou_thres = 0.45
            selected_classes = None
            
            if model:
                # Conf/IOU
                if "sam" not in model_name.lower():
                    conf_thres = st.slider("置信度 (Conf)", 0.0, 1.0, 0.25, 0.05, key=f"conf_{task_type}")
                    iou_thres = st.slider("IOU (NMS)", 0.0, 1.0, 0.45, 0.05, key=f"iou_{task_type}")
                
                # Class Filter
                # Class Filter
                selected_classes = None
                if hasattr(model, 'names') and "sam" not in model_name.lower():
                     selected_classes = render_class_filter(model, f"task_{task_type}")
                         
            show_conf = st.toggle("显示置信度", True, key=f"sc_{task_type}")
            show_labels = st.toggle("显示标签", True, key=f"sl_{task_type}")
            show_boxes = st.toggle("显示边框", True, key=f"sb_{task_type}")
            
            # Segmentation Option
            show_masks = True
            show_contours = False
            if "seg" in getattr(model, 'task', '') or "seg" in model_name:
                show_masks = st.toggle("显示掩码 (Masks)", True, key=f"sm_{task_type}")
                show_contours = st.toggle("显示轮廓 (External Contours)", False, key=f"s_cnt_{task_type}")

    with c_res:
        st.subheader("处理结果 (Results)")
        
        # Determine if input is Image or Video
        is_video = False
        if input_source_path:
            ext = os.path.splitext(input_source_path)[1].lower()
            if ext in ['.mp4', '.avi', '.mov']:
                is_video = True
        
        # --- Logic for Webcam ---
        if is_stream:
            run_Cam = st.checkbox("启动摄像头", key=f"cam_{task_type}")
            if run_Cam:
                st_frame = st.empty()
                cap = cv2.VideoCapture(0)
                while run_Cam and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    # Inference via Manager
                    if "sam" in model_name.lower():
                        frame_res = manager.predict(model, frame, prompts=prompts)
                    else:
                        frame_res = manager.track(model, frame, conf=conf_thres, iou=iou_thres, classes=selected_classes)
                    
                    # Plot via Manager
                    # Plot via Manager
                    res_rgb = manager.plot_result(
                        frame_res[0], 
                        show_conf, 
                        show_labels, 
                        show_boxes, 
                        show_masks=show_masks if 'show_masks' in locals() else True,
                        show_contours=show_contours if 'show_contours' in locals() else False
                    )
                    if res_rgb is not None:
                        st_frame.image(res_rgb, caption="Webcam", use_column_width=True)
                cap.release()
                
        # --- Logic for Video File ---
        elif is_video and input_source_path:
            st.video(input_source_path)
            if st.button("🚀 开始分析视频", key=f"vid_btn_{task_type}"):
                 vf = cv2.VideoCapture(input_source_path)
                 st_frame = st.empty()
                 stop_btn = st.button("🛑 停止", key=f"stop_{task_type}")
                 while vf.isOpened() and not stop_btn:
                     ret, frame = vf.read()
                     if not ret: break
                     
                     if "sam" in model_name.lower():
                         r = manager.predict(model, frame, prompts=prompts)
                     else:
                         r = manager.track(model, frame, conf=conf_thres, iou=iou_thres, classes=selected_classes)
                     
                     res_rgb = manager.plot_result(r[0], show_conf, show_labels, show_boxes)
                     if res_rgb is not None:
                         st_frame.image(res_rgb, caption="Video Processing", use_column_width=True)
                 vf.release()

        # --- Logic for Static Image ---
        elif input_source_path:
            # Read Image
            img_bgr = cv2.imread(input_source_path)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                c1, c2 = st.columns(2)
                with c1: st.image(img_rgb, caption="Original", use_column_width=True)
                
                # Cache Logic (Inference Only)
                # Key depends on: Model, Input, Conf, IOU, Classes, Prompts
                # Viz params (show_conf, etc) do NOT affect this key.
                input_hash = hash(input_source_path + str(os.path.getmtime(input_source_path)))
                inference_key = f"{model_name}_{input_hash}_{conf_thres}_{iou_thres}_{selected_classes}_{str(prompts)}"
                
                if 'yolo_inference_cache' not in st.session_state: st.session_state.yolo_inference_cache = {}
                
                start_btn = st.button("🚀 开始分析", type="primary", key=f"img_btn_{task_type}")
                
                results = None
                speed = None
                
                # Check Cache or Run Inference
                # Logic: If user clicked Start before (Active), AUTO-RERUN on param change (Cache Miss).
                
                # Active State Tracking
                active_key = f"active_file_{task_type}"
                if start_btn:
                    st.session_state[active_key] = input_source_path
                
                is_active = st.session_state.get(active_key) == input_source_path
                
                # Debug Panel Removed (v50 Cleanup)

                if is_active or inference_key in st.session_state.yolo_inference_cache:
                    if inference_key in st.session_state.yolo_inference_cache:
                         st.info("ℹ️ 读取缓存结果 (Cache Hit)")
                         results, speed = st.session_state.yolo_inference_cache[inference_key]
                    else:
                         # Active but Cache Miss -> Param Changed -> Re-run
                         st.toast(f"⚙️ 参数变更，正在重新分析... (Conf={conf_thres}, IOU={iou_thres})")
                         st.warning("⚠️ 参数变更，触发重算 (Re-running)...") # Visible feedback
                         with st.spinner("参数变更，重新分析中..."):
                             results = manager.predict(model, img_rgb, conf=conf_thres, iou=iou_thres, classes=selected_classes, prompts=prompts)
                             if results:
                                 speed = getattr(results[0], 'speed', None)
                                 st.session_state.yolo_inference_cache[inference_key] = (results, speed)
                
                if results:
                    st.caption(f"⚙️ 当前配置: Conf={conf_thres:.2f} | IOU={iou_thres:.2f} | Classes={len(selected_classes) if selected_classes else 'All'}")
                    
                    # Visualization (Runs every time params change, instant)
                    # Pass show_masks only if it exists in kwargs
                    res_rgb = manager.plot_result(
                        results[0], 
                        show_conf, 
                        show_labels, 
                        show_boxes, 
                        show_masks=show_masks if 'show_masks' in locals() else True,
                        show_contours=show_contours if 'show_contours' in locals() else False
                    )

                    # Quick Fix: Swap channel back because Manager inverted it (assuming BGR input)
                    if res_rgb is not None:
                        res_rgb = cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR)
                    
                    with c2:
                        if res_rgb is not None:
                            st.image(res_rgb, caption="Result", use_column_width=True)
                        if speed:
                             st.caption(f"⚡ Speed: Pre {speed.get('preprocess',0):.1f}ms | Inf {speed.get('inference',0):.1f}ms | Post {speed.get('postprocess',0):.1f}ms")
                    
                    # Data
                    with st.expander("检测数据"):
                         if "sam" not in model_name.lower() and hasattr(results[0], 'boxes') and results[0].boxes:
                             data = []
                             for box in results[0].boxes:
                                 data.append({
                                     "Class": results[0].names[int(box.cls)],
                                     "Conf": float(box.conf),
                                     "Box": box.xyxy.tolist()[0]
                                 })
                             st.dataframe(pd.DataFrame(data))
                else:
                    with c2:
                         # Placeholder
                         st.image(np.zeros_like(img_rgb), caption="Waiting...", use_column_width=True, clamp=True)
                         st.info("👈 请点击按钮开始 [分析]")
            else:
                st.error("无法读取图片文件")


def render_class_filter(model, key_prefix):
    """
    Unified Class Filter UI with Real-time Options Filtering.
    """
    if not hasattr(model, 'names') or not model.names:
        return None

    # CSS for compact buttons
    st.markdown("""
        <style>
        div[data-testid="column"] button {
            font-size: 0.8rem !important;
            padding: 0.2rem 0.6rem !important;
            min-height: 28px !important;
            height: auto !important;
        }
        </style>
    """, unsafe_allow_html=True)
        
    all_names = model.names
    options = list(all_names.values())
    session_key = f"{key_prefix}_classes"
    
    # Init default
    if session_key not in st.session_state:
        st.session_state[session_key] = options

    st.markdown("##### 🔍 类别筛选")
    
    # 1. Controls Row
    c1, c2, c3 = st.columns([3, 1, 1])
    
    with c1:
        kw = st.text_input("关键字", key=f"{key_prefix}_kw", label_visibility="collapsed", placeholder="输入关键字筛选选项...")
        
    with c2:
        # Select All
        if st.button("全选", key=f"{key_prefix}_all", help="选中所有类别"):
            st.session_state[session_key] = options
            st.rerun()
            
    with c3:
        if st.button("清空", key=f"{key_prefix}_clear"):
            st.session_state[session_key] = []
            st.rerun()
            
    # 2. Dynamic Options Logic
    # Filter options based on keyword, BUT keep currently selected items visible
    display_options = options
    if kw:
        matches = [n for n in options if kw.lower() in n.lower()]
        current_sel = st.session_state.get(session_key, [])
        # Union of Matches and Current Selection (so selected don't disappear)
        display_options = list(set(matches + [x for x in current_sel if x in options]))
        # Sort to keep order consistent with original options
        display_options.sort(key=lambda x: options.index(x))

    selected_names = st.multiselect("包含类别", display_options, key=session_key, label_visibility="collapsed")
    
    # Optimization Removed: Always return explicit list to aid debugging
    # if len(selected_names) == len(list(all_names.values())):
    #    return None
        
    return [k for k, v in all_names.items() if v in selected_names]

def render_yolo_tracking_content(manager):
    import cv2
    import os
    
    c_conf, c_res = st.columns([1, 2])
    
    with c_conf:
        st.markdown("#### 1. 输入源 (Input)")
        
        source_type = st.radio("选择输入", ["📷 摄像头 (Webcam)", "🎞️ 视频文件 (Video)"], key="track_src")
        
        input_source_path = None
        is_stream = False
        
        if source_type == "📷 摄像头 (Webcam)":
            is_stream = True
        else:
            uploaded_file = st.file_uploader("上传视频", type=['mp4', 'avi', 'mov'], key="track_up")
            if uploaded_file:
                upload_dir = "images/uploads"
                if not os.path.exists(upload_dir): os.makedirs(upload_dir, exist_ok=True)
                file_path = os.path.join(upload_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                input_source_path = file_path
        
        st.markdown("#### 2. 参数设置 (Params)")
        
        with st.container(height=500):
            # Model
            model_opts = ["yolo11n.pt", "yolo26n.pt", "yolo26s.pt", "yolov8n.pt", "yolo11n-seg.pt", "yolo11n-pose.pt"]
            model_name = st.selectbox("选择模型", model_opts, key="track_mdl")
            
            # Tracker
            tracker_type = st.radio("追踪算法", ["bytetrack.yaml", "botsort.yaml"], key="track_algo")
            st.caption("ByteTrack: 高效轻量; BoT-SORT: 鲁棒性更强")
            
            # Load Model
            with st.spinner(f"加载模型 {model_name}..."):
                model = manager.load_model(model_name)
                
            if model:
                # Unified Class Filter
                selected_classes = render_class_filter(model, "track")
            else:
                selected_classes = None
    
            conf_thres = st.slider("置信度", 0.0, 1.0, 0.25, 0.05, key="track_conf")
            iou_thres = st.slider("IOU", 0.0, 1.0, 0.45, 0.05, key="track_iou")
            
            show_conf = st.toggle("Conf", False, key="track_sc")
            show_boxes = st.toggle("Box", True, key="track_sb")
            show_trails = st.toggle("Trails (轨迹)", True, key="track_trails")
            
            # Check if model supports segmentation
            is_seg = False
            if hasattr(model, 'task') and model.task == 'segment': is_seg = True
            if "seg" in model_name: is_seg = True
            
            show_masks = True
            if is_seg:
                 show_masks = st.toggle("Masks", True, key="track_sm")
                 show_contours = st.toggle("Contours", False, key="track_scnt")
            else:
                 show_contours = False
            
            # 2. Count & Speed (Interactive) - REMOVED (v84)
            enable_count = False
            enable_speed = False
            
            # Init History
            if 'track_history' not in st.session_state:
                st.session_state.track_history = {}

    with c_res:
        st.subheader("追踪结果")
        
        # Helper to process frame
        def process_tracking_frame(frame, model, manager, classes=None, show_masks=True, show_contours=False):
            # 1. Track
            results = manager.track(model, frame, conf=conf_thres, iou=iou_thres, tracker=tracker_type, classes=classes)
            
            # 2. Plot Basic (Boxes + Trails)
            res_rgb = manager.plot_result(
                results[0], 
                show_conf=show_conf, 
                show_boxes=show_boxes,
                show_masks=show_masks,
                show_contours=show_contours,
                track_history=st.session_state.track_history if show_trails else None
            )
            
            if res_rgb is None: return frame
            
            # 3. Advanced Features (Count/Speed) - REMOVED (v84)
            # enable_count and enable_speed are Forced False above.
            
            # Final Output (BGR -> RGB)
            # Convert RGB back to BGR for solutions (which expect BGR usually)
            # Actually if we don't process solutions, we can just return res_rgb!
            # But wait, plot_result returns RGB.
            # If we don't do anything else, we just return res_rgb.
            
            return res_rgb


        if is_stream:
            run_btn = st.checkbox("启动摄像头追踪", key="track_run_cam")
            if run_btn and model:
                st_frame = st.empty()
                cap = cv2.VideoCapture(0)
                
                # Reset counters on start
                if 'counter_obj' in st.session_state: del st.session_state['counter_obj']
                if 'speed_obj' in st.session_state: del st.session_state['speed_obj']
                
                while run_btn and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    final_img = process_tracking_frame(frame, model, manager, classes=selected_classes, show_masks=show_masks, show_contours=show_contours)
                    st_frame.image(final_img, caption=f"Tracking: {tracker_type}", use_column_width=True)
                cap.release()
                
        elif input_source_path and model:
            st.video(input_source_path)
            if st.button("🚀 开始视频追踪", key="track_run_vid"):
                # Clear history for new run
                st.session_state.track_history = {}
                # Reset counters
                if 'counter_obj' in st.session_state: del st.session_state['counter_obj']
                if 'speed_obj' in st.session_state: del st.session_state['speed_obj']
                
                vf = cv2.VideoCapture(input_source_path)
                st_frame = st.empty()
                stop_btn = st.button("🛑 停止", key="track_stop")
                
                while vf.isOpened() and not stop_btn:
                    ret, frame = vf.read()
                    if not ret: break
                    
                    final_img = process_tracking_frame(frame, model, manager, classes=selected_classes, show_masks=show_masks, show_contours=show_contours)
                    st_frame.image(final_img, caption=f"Tracking: {tracker_type}", use_column_width=True)
                vf.release()

# Case Showcase Removed per User Request (v67)

def render_sam_lab():
    import uuid
    import json
    
    # Lazy Import
    try:
        from src.sam_manager import SAMManager
        from streamlit_drawable_canvas import st_canvas
        from PIL import Image
        import numpy as np
        import cv2
    except ImportError:
        st.error("无法导入 SAMManager 或 streamlit-drawable-canvas。请检查依赖。")
        return

    # --- 1. Initialization & State ---
    if 'sam_manager' not in st.session_state:
        st.session_state.sam_manager = SAMManager(model_path="sam3_hvit_b.pt")
        st.session_state.sam_manager.load_model()
        
    manager = st.session_state.sam_manager
    
    if 'sam_categories' not in st.session_state:
        st.session_state.sam_categories = [
            {"id": 1, "name": "object", "color": "#FF0000"},
            {"id": 2, "name": "background", "color": "#000000"}
        ]
    
    if 'sam_annotations' not in st.session_state:
        st.session_state.sam_annotations = {} 
        
    if 'sam_current' not in st.session_state:
        st.session_state.sam_current = {
            "prompts": [], 
            "bboxes": [],
            "mask": None,
            "image_path": None,
            "active_cat_id": 1
        }

    # Split Layout: Controls (1/4), Tabs (3/4)
    c_ctrl, c_main = st.columns([1, 3])
    
    with c_ctrl:
        st.subheader("1. 图像源")
        input_source = st.radio("来源", ["📁 Local Dir", "⬆️ Upload"], horizontal=True, key="sam_src_mode", label_visibility="collapsed")
        
        img_path, img_bytes = None, None
        if "Local" in input_source:
            base_dir = st.text_input("路径", value="./images", key="sam_base_dir")
            if os.path.exists(base_dir):
                files = [f for f in os.listdir(base_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                if files:
                    sel_file = st.selectbox("选择图片", files, key="sam_file_sel")
                    img_path = os.path.join(base_dir, sel_file)
                else: st.warning("无图片")
            else: st.error("路径不存在")
        else:
             up_file = st.file_uploader("上传图片", type=['jpg','png'], key="sam_up_file")
             if up_file:
                 img_bytes = up_file.getvalue()
                 img_path = f"upload_{up_file.name}"

        current_img = None
        if img_path:
            if st.session_state.sam_current["image_path"] != img_path:
                st.session_state.sam_current.update({
                    "prompts": [], "bboxes": [], "mask": None, "image_path": img_path, "cache_img": None
                })
            
            if st.session_state.sam_current.get("cache_img") is None:
                if img_bytes:
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    data = cv2.imdecode(nparr, 1)
                    if data is not None: st.session_state.sam_current["cache_img"] = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                elif os.path.exists(img_path):
                    data = cv2.imread(img_path)
                    if data is not None: st.session_state.sam_current["cache_img"] = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            current_img = st.session_state.sam_current.get("cache_img")

        st.divider()
        st.subheader("2. 类别管理")
        
        # Add New Category
        with st.expander("➕ 添加新类别", expanded=False):
            new_cat_name = st.text_input("类别名称", key="new_cat_name")
            new_cat_color = st.color_picker("类别颜色", "#00FF00", key="new_cat_color")
            if st.button("确认添加", use_container_width=True):
                if new_cat_name:
                    new_id = max([c["id"] for c in st.session_state.sam_categories] or [0]) + 1
                    st.session_state.sam_categories.append({"id": new_id, "name": new_cat_name, "color": new_cat_color})
                    st.success(f"已添加类别: {new_cat_name}")
                    st.rerun()

        # List and Edit Categories
        cats = st.session_state.sam_categories
        for i, cat in enumerate(cats):
            col1, col2, col3 = st.columns([3, 1, 1])
            # Category Name (Editable)
            new_name = col1.text_input(f"Name {cat['id']}", value=cat['name'], key=f"cat_name_{cat['id']}", label_visibility="collapsed")
            if new_name != cat['name']:
                st.session_state.sam_categories[i]['name'] = new_name
            
            # Color Picker
            new_color = col2.color_picker(f"Color {cat['id']}", value=cat['color'], key=f"cat_color_{cat['id']}", label_visibility="collapsed")
            if new_color != cat['color']:
                st.session_state.sam_categories[i]['color'] = new_color
            
            # Delete Button (Don't delete if only one left)
            if len(cats) > 1:
                if col3.button("🗑️", key=f"del_cat_{cat['id']}"):
                    st.session_state.sam_categories.pop(i)
                    st.rerun()
        
        st.divider()
        # Current Active category selector
        cat_names = [c["name"] for c in st.session_state.sam_categories]
        sel_cat_name = st.selectbox("当前工作类别", cat_names, key="sam_cat_sel")
        sel_cat = next((c for c in st.session_state.sam_categories if c["name"] == sel_cat_name), st.session_state.sam_categories[0])
        st.session_state.sam_current["active_cat_id"] = sel_cat["id"]

        st.divider()
        st.subheader("3. 标注列表")
        if img_path and img_path in st.session_state.sam_annotations:
            annos = st.session_state.sam_annotations[img_path]
            for i, ann in enumerate(annos):
                c_name = next((c["name"] for c in st.session_state.sam_categories if c["id"] == ann["category_id"]), "Unknown")
                k1, k2, k3 = st.columns([3, 1, 1])
                k1.markdown(f"**{i+1}. {c_name}**")
                if k2.button("👁️", key=f"focus_{i}"):
                    st.session_state.sam_current["focused_id"] = ann["id"]
                    st.rerun()
                if k3.button("✖️", key=f"del_{i}"):
                    del st.session_state.sam_annotations[img_path][i]
                    st.rerun()
        else: st.info("暂无标注")

    with c_main:
        t_inter, t_track, t_auto = st.tabs(["🎯 交互式标注", "🎞️ 目标追踪", "🤖 自动分割"])
        
        with t_inter:
            if current_img is None:
                st.info("👈 请先在左侧选择图片")
            else:
                # Sub-controls for Interactive (Aligning Label and Radio in columns)
                row_main = st.container()
                with row_main:
                    # Using more granular columns to keep things on "one line"
                    # Label, Tool, Label, Type, Spacer, Save, Clear
                    c_l1, c_r1, c_l2, c_r2, c_sp, c_b1, c_b2 = st.columns([0.4, 2.2, 0.4, 1.6, 0.1, 0.7, 0.7])
                    
                    with c_l1:
                        st.markdown("<p style='margin-top: 8px; font-weight: bold;'>工具:</p>", unsafe_allow_html=True)
                    with c_r1:
                        tool_mode = st.radio("工具", ["点击", "框选", "轮廓"], 
                                             horizontal=True, key="sam_tool_v2", label_visibility="collapsed")
                    
                    with c_l2:
                        st.markdown("<p style='margin-top: 8px; font-weight: bold;'>类型:</p>", unsafe_allow_html=True)
                    with c_r2: 
                         pt_type = st.radio("点类型", ["🟢 前景", "🔴 背景"], horizontal=True, key="sam_pt_type_v2", 
                                            label_visibility="collapsed", disabled=("框选" in tool_mode))
                         point_label = 0 if "背景" in pt_type else 1
                         stroke_color = "#FF0000" if point_label == 0 else "#0000FF"
                         if "框选" in tool_mode: stroke_color = "#00FF00"
                    
                    # c_sp is just a tiny spacer
                    
                    with c_b1:
                        # st.markdown("<div style='margin-top: 2px;'>", unsafe_allow_html=True)
                        if st.button("💾 保存", type="primary", use_container_width=True, key="sam_save_btn"):
                            if st.session_state.sam_current["mask"] is not None:
                                if img_path not in st.session_state.sam_annotations: 
                                    st.session_state.sam_annotations[img_path] = []
                                st.session_state.sam_annotations[img_path].append({
                                    "id": str(uuid.uuid4()), 
                                    "category_id": st.session_state.sam_current["active_cat_id"],
                                    "mask": st.session_state.sam_current["mask"], 
                                    "bbox": [0,0,0,0]
                                })
                                st.session_state.sam_current.update({"mask": None, "prompts": [], "bboxes": []})
                                st.rerun()

                    with c_b2:
                        # st.markdown("<div style='margin-top: 2px;'>", unsafe_allow_html=True)
                        if st.button("🧹 清除", use_container_width=True, key="sam_clear_interactive"):
                            st.session_state.sam_current.update({"mask": None, "prompts": [], "bboxes": []})
                            st.rerun()

                # Canvas Display
                display_img = current_img.copy()
                focused_id = st.session_state.sam_current.get("focused_id")
                if img_path and img_path in st.session_state.sam_annotations:
                    for ann in st.session_state.sam_annotations[img_path]:
                        is_focused = (focused_id == ann["id"])
                        alpha = 0.7 if is_focused else 0.2
                        c_col_hex = next((c["color"] for c in st.session_state.sam_categories if c["id"] == ann["category_id"]), "#00FF00")
                        c_col_rgb = tuple(int(c_col_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                        display_img = manager.overlay_mask(display_img, ann["mask"], color=c_col_rgb, alpha=alpha)

                if st.session_state.sam_current["mask"] is not None:
                    display_img = manager.overlay_mask(display_img, st.session_state.sam_current["mask"], color=(255, 255, 0), alpha=0.6)

                h_orig, w_orig = current_img.shape[:2]
                canvas_width = 1000 # Increased as requested
                canvas_height = int(h_orig * (canvas_width / w_orig))
                
                drawing_mode = "point"
                if "框选" in tool_mode: drawing_mode = "rect"
                elif "轮廓" in tool_mode: drawing_mode = "polygon"

                canvas_result = st_canvas(
                    fill_color="rgba(0, 0, 255, 0.1)", stroke_width=3, stroke_color=stroke_color,
                    background_image=Image.fromarray(display_img), update_streamlit=True,
                    height=canvas_height, width=canvas_width, drawing_mode=drawing_mode,
                    point_display_radius=8, key="sam_v2_canvas"
                )

                if canvas_result.json_data:
                    objects = canvas_result.json_data["objects"]
                    
                    # Filter out objects that are not relevant to the current tool mode
                    # For example, if tool_mode is "Point", only consider "point" objects.
                    # This prevents old shapes from other modes from interfering.
                    
                    filtered_objects = []
                    if "点击" in tool_mode:
                        filtered_objects = [obj for obj in objects if obj["type"] == "point"]
                    elif "框选" in tool_mode:
                        filtered_objects = [obj for obj in objects if obj["type"] == "rect"]
                    elif "轮廓" in tool_mode:
                        filtered_objects = [obj for obj in objects if obj["type"] == "polygon"]

                    if len(filtered_objects) > 0:
                        new_prompts, new_labels, new_bboxes = [], [], []
                        sx, sy = w_orig / canvas_width, h_orig / canvas_height
                        for obj in filtered_objects:
                            if obj["type"] == "point":
                                new_prompts.append([int(obj["left"] * sx), int(obj["top"] * sy)])
                                # Use the point_label determined by the radio button
                                new_labels.append(point_label) 
                            elif obj["type"] == "rect":
                                x, y = int(obj["left"] * sx), int(obj["top"] * sy)
                                w, h = int(obj["width"] * sx), int(obj["height"] * sy)
                                new_bboxes.append([x, y, x+w, y+h])
                            elif obj["type"] == "polygon":
                                # For Lasso, treat as a bounding box for simplicity with SAM
                                x, y = int(obj["left"] * sx), int(obj["top"] * sy)
                                w, h = int(obj["width"] * sx), int(obj["height"] * sy)
                                new_bboxes.append([x, y, x+w, y+h])

                        # Only run prediction if there are new inputs or inputs have changed
                        current_state_sig = {"prompts": new_prompts, "bboxes": new_bboxes}
                        prev_state_sig = {"prompts": st.session_state.sam_current["prompts"], "bboxes": st.session_state.sam_current["bboxes"]}

                        if current_state_sig != prev_state_sig:
                            res = manager.predict_image(current_img, points=new_prompts or None, labels=new_labels or None, bbox=new_bboxes or None)
                            if res and res[0].masks is not None:
                                masks = res[0].masks.data.cpu().numpy()
                                if len(masks) > 0:
                                    st.session_state.sam_current["mask"] = np.any(masks, axis=0)
                                    st.session_state.sam_current["prompts"] = new_prompts
                                    st.session_state.sam_current["bboxes"] = new_bboxes
                                    st.rerun()
                            else: # If no results, clear current mask
                                if st.session_state.sam_current["mask"] is not None:
                                    st.session_state.sam_current.update({"mask": None, "prompts": [], "bboxes": []})
                                    st.rerun()
                    elif st.session_state.sam_current["mask"] is not None: # No objects drawn, clear mask
                        st.session_state.sam_current.update({"mask": None, "prompts": [], "bboxes": []})
                        st.rerun()

        with t_track:
            st.subheader("🎞️ 视频目标追踪 (SAM 2/3)")
            st.info("追踪模式允许在视频帧中选择物体并自动追踪后续帧。")
            st.button("开始追踪演示", on_click=lambda: manager.track_video(None))

        with t_auto:
            st.subheader("🤖 全图自动分割")
            st.info("自动检测并分割图片中的所有显著物体。")
            if current_img is not None:
                if st.button("✨ 一键自动分割", key="sam_auto_run"):
                    # This will trigger SAM's automatic segmentation
                    with st.spinner("正在自动分割图片..."):
                        auto_results = manager.predict_image(current_img, points=None, labels=None, bbox=None, prompt_type="everything")
                        if auto_results and auto_results[0].masks is not None:
                            all_masks = auto_results[0].masks.data.cpu().numpy() # (N, H, W)
                            if len(all_masks) > 0:
                                if img_path not in st.session_state.sam_annotations: st.session_state.sam_annotations[img_path] = []
                                for i, mask in enumerate(all_masks):
                                    st.session_state.sam_annotations[img_path].append({
                                        "id": str(uuid.uuid4()), "category_id": st.session_state.sam_current["active_cat_id"],
                                        "mask": mask, "bbox": [0,0,0,0] # Bbox not directly from auto-seg
                                    })
                                st.success(f"成功分割 {len(all_masks)} 个物体！")
                                st.rerun()
                            else:
                                st.warning("未检测到任何物体。")
                        else:
                            st.error("自动分割失败。")
            else: st.warning("请先选择图片")


def inject_custom_css():
    st.markdown("""
        <style>
        /* Global Background & Glassmorphism */
        .stApp {
            background-color: #f8f9fa;
        }
        
        /* Card Style Container */
        .premium-card {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            margin-bottom: 20px;
            transition: transform 0.2s ease;
        }
        .premium-card:hover {
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.08);
        }

        /* Section Header Styling */
        .section-header {
            font-size: 1.2rem;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
        }
        
        /* Live Badge for Capture */
        .live-badge {
            background-color: #ef4444;
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: bold;
            text-transform: uppercase;
            margin-left: 10px;
            animation: pulse-red 2s infinite;
        }
        
        @keyframes pulse-red {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        /* Face Sample Image Styling */
        .face-sample-container img {
            border-radius: 8px;
            border: 2px solid #e5e7eb;
            transition: all 0.3s ease;
            cursor: zoom-in;
        }
        .face-sample-container img:hover {
            border-color: #3b82f6;
            transform: scale(1.05);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }
        
        /* Button Polish */
        .stButton button {
            border-radius: 8px !important;
        }
        
        /* Sidebar List Simulation */
        .sidebar-list-item {
            padding: 8px 12px;
            border-radius: 6px;
            margin-bottom: 4px;
            background: rgba(255,255,255,0.4);
            border: 1px solid transparent;
            cursor: pointer;
        }
        .sidebar-list-item:hover {
            background: rgba(59, 130, 246, 0.05);
            border: 1px solid rgba(59, 130, 246, 0.2);
        }
        
        /* Sidebar Button List Styling (Cleaned) */
        div[data-testid="stVerticalBlock"] > div.element-container > div.stButton > button {
            text-align: left !important;
            width: 100% !important;
            justify-content: flex-start !important;
            border: 1px solid transparent !important;
            transition: all 0.2s !important;
        }
        </style>
    """, unsafe_allow_html=True)

def render_face_lab():
    try:
        from src.face_manager import FaceManager
        import cv2
        import numpy as np
        from PIL import Image
        
        # Initialize Manager (Force reset if old version detected in session_state)
        # Initialize Manager (Force reset if old version detected in session_state)
        # Bumping version to v113 to force reload new gesture methods
        if 'face_manager_v113' not in st.session_state:
            st.session_state.face_manager_v113 = FaceManager()
        manager = st.session_state.face_manager_v113
        
    except ImportError as e:
        st.error(f"依赖缺失或模块加载失败: {e}")
        return

    inject_custom_css()
    # st.header("👤 人脸分析实验室 (Face Analysis Lab)")
    
    tab1, tab2, tab3 = st.tabs(["🖼️ 人脸分析 (Analysis)", "🗃️ 人脸库 (Face Bank)", "✋ 手势库 (Gesture Bank)"])
    
    with tab1:
        c_conf, c_res = st.columns([1, 2])
        
        with c_conf:
            st.markdown("#### 1. 输入源 (Input)")
            source_type = st.radio("选择输入", ["🖼️ 图片上传", "📷 摄像头 (Webcam)"], key="face_src")
            
            input_img = None
            is_webcam = False
            
            if source_type == "🖼️ 图片上传":
                up = st.file_uploader("上传图片", type=['jpg', 'png', 'jpeg'], key="face_up")
                if up:
                    input_img = Image.open(up)
                    input_img = np.array(input_img.convert('RGB'))
                    input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
            else:
                is_webcam = True
                
            st.markdown("#### 2. 参数配置 (Parameters)")
            do_detect = st.toggle("人脸检测 (Detection)", True, key="face_det")
            face_mesh_mode = st.radio("人脸关键点 (Landmarks)", ["不显示", "基础 (5点)", "稠密 (478点)"], index=0, key="face_mesh_mode")
            do_recog = st.toggle("人脸识别 (Recognition)", False, key="face_rec")
            
            st.markdown("#### 3. 手势分析 (Gesture)")
            do_hand_mesh = st.toggle("显示手势骨架 (Hand Mesh)", False, key="hand_mesh")
            do_gesture = st.toggle("手势识别 (Gestures)", False, key="hand_gest")
            
            with st.expander("🛠️ 详细参数", expanded=False):
                det_threshold = st.slider("检测阈值 (Detector)", 0.0, 1.0, 0.5, 0.05)
                rec_threshold = st.slider("识别阈值 (Recognition)", 0.0, 1.0, 0.6, 0.05)
                auto_save = st.toggle("自动保存陌生人", False, help="识别为 Unknown 时自动保存到人脸库")
            

        with c_res:
            st.markdown("#### 💡 分析结果")
            if is_webcam:
                run_webcam = st.checkbox("启动摄像头", key="face_cam_run")
                if run_webcam:
                    st_frame = st.empty()
                    cap = cv2.VideoCapture(0)
                    while run_webcam:
                        ret, frame = cap.read()
                        if not ret: break
                        
                        processed_frame, is_threat = process_face_frame(
                            frame, manager, do_detect, face_mesh_mode, do_recog, 
                            rec_threshold, auto_save, do_hand_mesh, do_gesture
                        )
                        if is_threat:
                            processed_frame = manager.apply_red_glow(processed_frame)
                        
                        st_frame.image(processed_frame, channels="BGR", use_column_width=True)
                        
                        # Store current frame for capture feature
                        st.session_state.current_face_frame = frame.copy()
                        
                        # Stop check
                        if not st.session_state.get("face_cam_run", False):
                            break
                    cap.release()
            elif input_img is not None:
                processed, _ = process_face_frame(
                    input_img, manager, do_detect, face_mesh_mode, do_recog, 
                    rec_threshold, False, do_hand_mesh, do_gesture
                )
                st.image(processed, channels="BGR", use_column_width=True)
            else:
                st.info("请在左侧上传图片或启动摄像头")

    with tab2:
        render_face_bank_management(manager)

    with tab3:
        render_gesture_bank_management(manager)

def process_face_frame(frame, manager, do_detect, face_mesh_mode, do_recog, threshold, auto_save, do_hand_mesh, do_gesture):
    import cv2
    import os
    
    faces = []
    face_landmarks = None
    names = []
    hands = None
    
    # 1. Face Process
    if do_detect or do_recog:
        faces = manager.detect_faces(frame)
        
    if face_mesh_mode == "稠密 (478点)":
        face_landmarks = manager.get_face_landmarks(frame)
    elif face_mesh_mode == "基础 (5点)":
        # If faces detected, extract basic landmarks if available in face dict
        face_landmarks = [f.get('basic_landmarks') for f in faces if f.get('basic_landmarks')]
        if not face_landmarks:
             # Fallback if detector didn't provide points
             pass

    if do_recog and faces:
        for face in faces:
            x, y, w, h = face['bbox']
            crop = frame[max(0,y):y+h, max(0,x):x+w]
            if crop.size == 0: 
                names.append(("Unknown", 1.0))
                continue
            
            emb = manager.get_embedding(crop)
            if emb is not None:
                name, dist = manager.recognize_face(emb, threshold=threshold)
                names.append((name, dist))
                
                if name == "Unknown" and auto_save:
                    strangers_dir = os.path.join(manager.face_bank_path, "Strangers")
                    os.makedirs(strangers_dir, exist_ok=True)
                    person_id = f"Stranger_{len(os.listdir(strangers_dir)) + 1}"
                    manager.save_new_face(crop, person_id)
            else:
                names.append(("Unknown", 1.0))
    
    # 2. Hand Process
    if do_hand_mesh or do_gesture:
        hands = manager.get_hand_analysis(frame, do_landmarks=do_hand_mesh, do_gesture=do_gesture)
        if hands:
            # Store first detected hand for custom recording
            st.session_state.current_hand_landmarks = hands[0]['landmarks']
                
    # 3. Draw Results
    res_img, is_threat = manager.draw_results(
        frame.copy(), 
        faces=faces if (do_detect or do_recog) else None,
        face_landmarks=face_landmarks,
        recognized_names=names if do_recog else None,
        hands=hands
    )
        
    return res_img, is_threat

def render_gesture_bank_management(manager):
    import cv2
    import mediapipe as mp
    import time

    st.subheader("✋ 手势定义与预警配置")
    st.info("在这里您可以定义系统内置手势触发的特定含义。")
    
    gmap = manager.load_gesture_map()
    
    # Display table for mapping
    # Hand types in MediaPipe: Thumb_Up, Thumb_Down, Victory, OK, Closed_Fist, Open_Palm, Pointing_Up
    std_gestures = ["Thumb_Up", "Thumb_Down", "Victory", "OK", "Closed_Fist", "Open_Palm", "Pointing_Up", "ILoveYou"]
    
    new_map = {}
    cols = st.columns(2)
    for i, gest in enumerate(std_gestures):
        with cols[i % 2]:
            current_val = gmap.get(gest, gest)
            new_val = st.text_input(f"手势: {gest} 的含义", value=current_val, key=f"gest_{gest}")
            new_map[gest] = new_val
            
    if st.button("💾 保存标准手势配置", use_container_width=True):
        manager.save_gesture_map(new_map)
        st.success("配置已更新")

    st.divider()
    st.divider()
    st.subheader("📸 录制新手势 (Custom Recognition)")
    st.info("启动录制模式后，摆出该手势，点击【立即录制】即可保存。")

    # Control Columns: Library (Left) | Recorder (Right)
    c_list, c_rec = st.columns([1, 1.5])
    
    with c_list:
        st.markdown("#### Custom Gesture Library")
        st.caption("Here you can manage your recorded gestures and their meanings.")
        
        if not manager.custom_gestures:
             st.info("暂无自定义手势")
        else:
            # Scrollable container
            with st.container(height=400, border=True):
                # We need to save changes to meanings
                updated_meanings = {}
                has_changes = False
                
                for cg in list(manager.custom_gestures.keys()):
                    st.markdown(f"**✋ {cg}**")
                    
                    # 1. Meaning Input
                    current_meaning = gmap.get(cg, cg)
                    new_meaning = st.text_input(f"Meaning ({cg})", value=current_meaning, key=f"mean_{cg}", label_visibility="collapsed")
                    
                    if new_meaning != current_meaning:
                        updated_meanings[cg] = new_meaning
                        has_changes = True

                    # 2. Delete Button
                    if st.button("🗑️ Delete", key=f"del_g_{cg}"):
                        del manager.custom_gestures[cg]
                        manager.save_custom_gestures()
                        # Also remove from map? Optional, but cleaner.
                        # if cg in gmap: del gmap[cg]; manager.save_gesture_map(gmap)
                        st.rerun()
                    
                    st.divider()
                
                # Global Save for Meanings (if any changed in this loop)
                # Actually, better to have a single Save button at bottom of list
                if has_changes:
                     if st.button("💾 保存所有含义修改", key="save_custom_meanings", type="primary"):
                         for k, v in updated_meanings.items():
                             gmap[k] = v
                         manager.save_gesture_map(gmap)
                         st.success("含义已更新")
                         time.sleep(0.5)
                         st.rerun()

    with c_rec:
        st.markdown("#### 📹 Record New Gesture")
        
        # Name Input
        new_gest_name = st.text_input("Name (e.g., Finger_Heart)", key="new_gest_name_input")
        
        # Toggle Recording Mode
        run_recording = st.toggle("🔴 启动录制模式 (Start Camera)", key="toggle_gesture_rec")
        
        # Capture Button (Always visible but active only if running)
        # We use a callback or session state logic.
        # Logic: Loop updates session_state['latest_hand']. Button reads it.
        if st.button("⏺️ 立即录制当前手势 (Capture)", type="primary", use_container_width=True, disabled=not run_recording):
            if not new_gest_name:
                st.warning("请先输入手势名称")
            else:
                last_lms = st.session_state.get("latest_rec_hand_landmarks")
                if last_lms:
                    manager.save_custom_gesture(last_lms, new_gest_name)
                    
                    # Also add to gesture map with default Name as Meaning
                    if new_gest_name not in gmap:
                        gmap[new_gest_name] = new_gest_name
                        manager.save_gesture_map(gmap)
                        
                    st.success(f"已录制: {new_gest_name}")
                    st.rerun()
                else:
                    st.error("未检测到手势，请确保手部在画面中")

        # Video Loop Area
        rec_placeholder = st.empty()
        
        if run_recording:
             cap = cv2.VideoCapture(0)
             # Warmup
             cap.read()
             
             with rec_placeholder:
                 st_display = st.image([])
                 
                 while run_recording:
                     ret, frame = cap.read()
                     if not ret:
                         time.sleep(0.1)
                         continue
                         
                     # Detect Hands
                     # We reuse the manager's detection logic but lightweight
                     # Hand Landmarker expects RGB
                     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                     
                     # Async detect or sync? Sync is easier for loop
                     # FaceManager init_mediapipe_tasks initializes self.hand_landmarker
                     if manager.hand_landmarker:
                         detection_result = manager.hand_landmarker.detect(mp_image)
                         
                         draw_frame = frame.copy()
                         
                         if detection_result.hand_landmarks:
                             # Get first hand
                             hand_lms = detection_result.hand_landmarks[0]
                             
                             # Update Session State for Capture logic
                             st.session_state.latest_rec_hand_landmarks = hand_lms
                             
                             # Draw Landmarks manually or assume FaceManager has a helper
                             # We'll do a quick draw here for feedback
                             for lm in hand_lms:
                                 h, w, c = frame.shape
                                 cx, cy = int(lm.x * w), int(lm.y * h)
                                 cv2.circle(draw_frame, (cx, cy), 5, (0, 255, 0), -1)
                             
                             # Connect outlines (simple)
                             # Wrist (0) is key
                             cv2.putText(draw_frame, "HAND DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                         else:
                             st.session_state.latest_rec_hand_landmarks = None
                             cv2.putText(draw_frame, "NO HAND", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                             
                         st_display.image(draw_frame, channels="BGR", caption="实时预览")
                     else:
                         st.error("Hand Landmarker failed to initialize.")
                         break
                         
                     # Limit FPS
                     time.sleep(0.03)

def render_face_bank_management(manager):
    import cv2 # Local import to ensure it's available
    import time
    import shutil # Added missing import
    if not os.path.exists(os.path.join(manager.face_bank_path, "Strangers")):
        os.makedirs(os.path.join(manager.face_bank_path, "Strangers"))
    st.markdown("### 👥 成员管理")

    # 1. Dialogs
    @st.dialog("📋 重命名成员")
    def rename_person_dialog(old_name):
        st.write(f"正在为人员 `{old_name}` 设置新姓名")
        new_name = st.text_input("请输入新姓名", value=old_name)
        c1, c2 = st.columns(2)
        if c1.button("💾 确认保存", use_container_width=True, type="primary"):
            if new_name and new_name != old_name:
                old_path = os.path.join(manager.face_bank_path, old_name)
                new_path = os.path.join(manager.face_bank_path, new_name)
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    if old_name in manager.blacklist:
                        idx = manager.blacklist.index(old_name)
                        manager.blacklist[idx] = new_name
                        manager.save_blacklist()
                    st.session_state.member_sel = new_name 
                    manager.load_face_bank()
                    st.rerun()
                else:
                    st.error("该姓名已存在")
            else:
                st.rerun() 
        if c2.button("🚫 取消", use_container_width=True):
            st.rerun()

    @st.dialog("➕ 新增成员")
    def add_member_dialog():
        # Use a form to prevent premature reruns and ensure state capture
        with st.form("add_member_form"):
            new_name = st.text_input("请输入新成员姓名")
            # Buttons inside form
            c1, c2 = st.columns(2)
            submitted = c1.form_submit_button("💾 确认新增", type="primary", use_container_width=True)
            cancelled = c2.form_submit_button("🚫 取消", use_container_width=True)
            
            if submitted:
                if new_name:
                    person_dir = os.path.join(manager.face_bank_path, new_name)
                    if not os.path.exists(person_dir):
                        os.makedirs(person_dir)
                        st.session_state.member_sel = new_name
                        manager.load_face_bank()
                        st.rerun()
                    else:
                        st.error("该成员已存在")
                else:
                    st.warning("姓名不能为空")
            
            if cancelled:
                st.rerun()

    @st.dialog("🔄 转移图片 (Transfer Image)")
    def transfer_dialog_func(img_path, current_category):
        # img_path: full path to the image
        # current_category: 'member', 'stranger', 'blacklist'
        
        st.write(f"正在转移图片: `{os.path.basename(img_path)}`")
        
        # 1. Select Target Category
        target_cat = st.radio("目标库 (Target Bank)", ["👤 正式成员", "🕵️ 陌生人", "🚫 黑名单"], horizontal=True)
        
        # 2. Select Target Person
        target_name = None
        
        if target_cat == "👤 正式成员":
            # List existing members + Create New
            options = ["➕ 新建成员 (Create New)"] + people
            sel = st.selectbox("选择目标成员", options)
            if sel == "➕ 新建成员 (Create New)":
                target_name = st.text_input("输入新成员姓名")
            else:
                target_name = sel
                
        elif target_cat == "🕵️ 陌生人":
            # List existing strangers (folders in Strangers)
            strangers_list = sorted([d for d in os.listdir(os.path.join(manager.face_bank_path, "Strangers")) if os.path.isdir(os.path.join(manager.face_bank_path, "Strangers", d))])
            options = ["➕ 新建陌生人ID (Create New)"] + strangers_list
            sel = st.selectbox("选择目标陌生人", options)
            if sel == "➕ 新建陌生人ID (Create New)":
                target_name = st.text_input("输入新ID (e.g. Stranger_99)")
            else:
                target_name = sel
                
        elif target_cat == "🚫 黑名单":
            # List blacklist
            options = ["➕ 新建黑名单 (Create New)"] + manager.blacklist
            sel = st.selectbox("选择目标黑名单", options)
            if sel == "➕ 新建黑名单 (Create New)":
                target_name = st.text_input("输入黑名单姓名")
            else:
                target_name = sel

        # Unique key for button to prevent state mix-up
        if st.button("💾 确认转移", type="primary", use_container_width=True, key=f"btn_tr_confirm_{abs(hash(img_path))}"):
            if target_name:
                # Debug info
                # st.write(f"Debug: Moving {img_path} -> {target_name}")
                
                # Determine target path
                final_target_dir = ""
                if target_cat == "👤 正式成员":
                    final_target_dir = os.path.join(manager.face_bank_path, target_name)
                    
                elif target_cat == "🕵️ 陌生人":
                    final_target_dir = os.path.join(manager.face_bank_path, "Strangers", target_name)
                    
                elif target_cat == "🚫 黑名单":
                    final_target_dir = os.path.join(manager.face_bank_path, target_name)
                    if target_name not in manager.blacklist:
                        manager.blacklist.append(target_name)
                        manager.save_blacklist()

                if not os.path.exists(final_target_dir):
                    os.makedirs(final_target_dir)
                
                # Move file
                full_target_path = os.path.join(final_target_dir, os.path.basename(img_path))
                
                if os.path.exists(img_path):
                    shutil.move(img_path, full_target_path)
                    st.toast(f"✅ 已成功转移至: {target_name}")
                    manager.load_face_bank()
                    time.sleep(0.5) # Wait for toast
                    st.rerun()
                else:
                    st.error(f"源文件不存在: {img_path}")
            else:
                st.warning("请输入或选择目标名称")

    @st.dialog("✏️ 重命名 (Rename)")
    def rename_dialog_func(current_name, category):
        # category: 'stranger', 'blacklist', 'member'
        st.write(f"正在重命名: `{current_name}`")
        new_name = st.text_input("请输入新名称", value=current_name)
        
        if st.button("💾 确认修改", type="primary", use_container_width=True):
            if new_name and new_name != current_name:
                # Logic depends on category
                if category == 'stranger':
                    old_p = os.path.join(manager.face_bank_path, "Strangers", current_name)
                    new_p = os.path.join(manager.face_bank_path, "Strangers", new_name)
                    # Use rename logic similar to button
                    if os.path.exists(old_p):
                        os.rename(old_p, new_p)
                        st.session_state.stranger_sel = new_name
                        st.success("修改成功")
                        st.rerun()
                        
                elif category == 'blacklist':
                    old_p = os.path.join(manager.face_bank_path, current_name)
                    new_p = os.path.join(manager.face_bank_path, new_name)
                    
                    if current_name in manager.blacklist:
                        manager.blacklist.remove(current_name)
                        manager.blacklist.append(new_name)
                        manager.save_blacklist()
                        
                    if os.path.exists(old_p):
                        os.rename(old_p, new_p)
                    
                    st.session_state.blacklist_sel = new_name
                    st.success("修改成功")
                    st.rerun()
                    
                elif category == 'member':
                     # We already have rename logic for members, can unify later
                     pass
            elif new_name == current_name:
                 st.info("名称未变更")
            else:
                 st.warning("名称不能为空")

    # 2. Data Prep
    people = sorted([p for p in os.listdir(manager.face_bank_path) 
                     if os.path.isdir(os.path.join(manager.face_bank_path, p)) 
                     and p != "Strangers" 
                     and p not in manager.blacklist])

    # 3. Layout Grid
    # Row 1: Content (List, Grid, Video)
    r1c1, r1c2, r1c3 = st.columns([1, 1.5, 1.5])
    
    # Row 2: Controls (Add, Actions, Save)
    r2c1, r2c2, r2c3 = st.columns([1, 1.5, 1.5])

    # --- Column 1: Member List ---
    with r1c1:
        st.markdown("**📋 成员列表**")
        with st.container(border=True, height=450):
            current_sel = st.session_state.get("member_sel", "-- 请选择 --")
            if st.button("-- 请选择 --", key="btn_reset", use_container_width=True, 
                         type="primary" if current_sel == "-- 请选择 --" else "secondary"):
                st.session_state.member_sel = "-- 请选择 --"
                st.rerun()

            for person in people:
                is_active = (current_sel == person)
                if st.button(f"👤 {person}", key=f"btn_{person}", use_container_width=True, 
                             type="primary" if is_active else "secondary"):
                    st.session_state.member_sel = person
                    st.rerun()
    
    with r2c1:
        # Move "Add Member" here as a dialog trigger
        if st.button("➕ 新增成员", use_container_width=True):
            add_member_dialog()

    # --- Column 2: Face Samples ---
    selected_person = st.session_state.get("member_sel", "-- 请选择 --")
    refresh_samples_func = None # Placeholder for the function

    with r1c2:
        st.markdown("**👤 当前库详情**")
        if selected_person != "-- 请选择 --":
            person_path = os.path.join(manager.face_bank_path, selected_person)
            
            # GHOST FIX: Use st.empty() for the wrapper, then container inside
            # But we need a border. st.container(border=True) is fixed.
            # To clear content inside a container, we can use a single st.empty() inside it 
            # and render everything into that empty block.
            with st.container(border=True, height=450):
                sample_grid_placeholder = st.empty() # The magic cleaner

                def refresh_samples(show_delete=True):
                    # This function redraws the entire grid clean every time
                    with sample_grid_placeholder.container():
                        imgs = sorted([f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                        if imgs:
                            cols = st.columns(3)
                            for i, img_name in enumerate(imgs):
                                with cols[i%3]:
                                    img_full_path = os.path.join(person_path, img_name)
                                    st.image(img_full_path, use_column_width=True)

                                    if show_delete:
                                        # Per-image controls (Popover)
                                        with st.popover("转移/删除", use_container_width=True):
                                            if st.button("↪️ 转移", key=f"tr_mem_{selected_person}_{img_name}", use_container_width=True):
                                                transfer_dialog_func(img_full_path, 'member')
                                            
                                            if st.button("🗑️ 删除", key=f"del_mem_{selected_person}_{img_name}", type="primary", use_container_width=True):
                                                manager.delete_face_sample(img_full_path)
                                                st.rerun()
                        else:
                            st.info("暂无人脸样本")
                
                refresh_samples_func = refresh_samples
                refresh_samples(show_delete=True)
        else:
            with st.container(border=True, height=450):
                st.info("👈 请在左侧选择一名成员")

    with r2c2:
        if selected_person != "-- 请选择 --":
            btn_cols = st.columns(4)
            with btn_cols[0]:
                if st.button("📝 改名", use_container_width=True):
                    rename_person_dialog(selected_person)
            with btn_cols[1]:
                # Toggle capture via session state, actual UI is in Col 3
                is_capturing = (st.session_state.get("capturing_for") == selected_person)
                btn_text = "⏹️ 停止采集" if is_capturing else "📸 采集"
                btn_type = "primary" if is_capturing else "secondary"
                
                if st.button(btn_text, use_container_width=True, type=btn_type):
                    if is_capturing:
                         st.session_state.capturing_for = None # Stop
                    else:
                         st.session_state.capturing_for = selected_person # Start
                    st.rerun()
            with btn_cols[2]:
                if st.button("🚨 删除", use_container_width=True, type="primary"):
                    shutil.rmtree(os.path.join(manager.face_bank_path, selected_person))
                    st.session_state.member_sel = "-- 请选择 --"
                    manager.load_face_bank()
                    st.rerun()
            with btn_cols[3]:
                if st.button("🚫 拉黑", use_container_width=True):
                    if selected_person not in manager.blacklist:
                        manager.blacklist.append(selected_person)
                        manager.save_blacklist()
                        st.session_state.member_sel = "-- 请选择 --"
                        st.warning(f"已拉黑")
                        st.rerun()

    # --- Column 3: Capture ---
    cap_target = st.session_state.get("capturing_for")
    
    with r1c3:
        st.markdown('**📸 采集预览**')
        # Placeholder for video to ensure it stays in Row 1
        video_placeholder = st.empty()
        
        # Default state when not capturing
        if not (cap_target and cap_target == selected_person):
             with video_placeholder.container():
                 with st.container(border=True, height=450):
                     st.info("点击下方“采集”按钮开启摄像头")

    save_btn_placeholder = r2c3.empty() # For the save button in Row 2

    # Logic for Capture Loop
    if cap_target and cap_target == selected_person:
        # 1. Render Save Button in Row 2
        with save_btn_placeholder:
             # Use a unique key to prevent duplicate ID errors
             # Key must be static to register click
             save_trigger = st.button("💾 保存当前人脸", key=f"save_btn_{selected_person}_persistent", use_container_width=True, type="primary")
        
        if save_trigger:
            st.session_state.should_save_face = True
        
        # 2. Render Video in Row 1
        cap = cv2.VideoCapture(0)
        
        # FIX 2: Camera Warmup to prevent black frames on startup/rerun
        for _ in range(5):
             cap.read()
        
        while True: # Infinite loop until stopped
            ret, frame = cap.read()
            if not ret: 
                time.sleep(0.1)
                continue
            
            # Draw Face Box
            faces = manager.detect_faces(frame)
            disp_frame = frame.copy()
            if faces:
                for f in faces:
                    bx = f['bbox']
                    cv2.rectangle(disp_frame, (bx[0], bx[1]), (bx[0]+bx[2], bx[1]+bx[3]), (0, 255, 0), 2)
            
            # Update Video Frame
            video_placeholder.image(disp_frame, channels="BGR")
            
            # Handle Save
            if st.session_state.get("should_save_face"):
                if faces:
                    f = max(faces, key=lambda x: x['bbox'][2]*x['bbox'][3])
                    bx = f['bbox']
                    # Bounds Check
                    y1, y2 = max(0,bx[1]), min(frame.shape[0], bx[1]+bx[3])
                    x1, x2 = max(0,bx[0]), min(frame.shape[1], bx[0]+bx[2])
                    crop = frame[y1:y2, x1:x2]
                    
                    if crop.size > 0:
                        manager.save_new_face(crop, cap_target)
                        st.toast(f"✅ 已保存样本！", icon="📸")
                        if refresh_samples_func:
                            refresh_samples_func(show_delete=False)
                    else:
                        st.error("无效图像区域")
                else:
                    st.warning("未检测到人脸")
                
                st.session_state.should_save_face = False
            
            # Break conditions
            # We check if the user switched person or clicked "Capture" again (which toggles it off in session state)
            # But the button press re-runs script, so this loop is naturally killed on rerun.
            # We just need to check if we should stop.
            if st.session_state.get("capturing_for") != selected_person:
                break
            
            # Small sleep to prevent UI lockup if needed, though Streamlit usually handles it.
            # time.sleep(0.01) 
        
        cap.release()

    st.markdown('</div>', unsafe_allow_html=True)

    # --- 2. 陌生人管理 (Stranger Management) ---
    st.markdown('<div class="premium-card"><div class="section-header">👤 陌生人管理</div>', unsafe_allow_html=True)
    
    strangers_path = os.path.join(manager.face_bank_path, "Strangers")
    if not os.path.exists(strangers_path): os.makedirs(strangers_path)
    
    strangers = sorted([d for d in os.listdir(strangers_path) if os.path.isdir(os.path.join(strangers_path, d))])
    
    if not strangers:
        st.info("暂无新捕获的陌生人。")
    else:
        # Layout: Left List, Right Details
        s_c1, s_c2 = st.columns([1, 2.5])
        
        # --- Column 1: Stranger List ---
        with s_c1:
            st.markdown("**🆔 陌生人列表**")
            with st.container(border=True, height=350):
                current_stranger = st.session_state.get("stranger_sel", None)
                if not current_stranger and strangers:
                    current_stranger = strangers[0]
                    st.session_state.stranger_sel = current_stranger

                for s_id in strangers:
                    is_active = (current_stranger == s_id)
                    if st.button(f"👤 {s_id}", key=f"btn_s_{s_id}", use_container_width=True, 
                                 type="primary" if is_active else "secondary"):
                        st.session_state.stranger_sel = s_id
                        st.rerun()
        
        # --- Column 2: Details & Actions ---
        with s_c2:
            st.markdown(f"**🕵️‍♂️ 详情与操作: `{current_stranger}`**")
            st_p_path = os.path.join(strangers_path, current_stranger) if current_stranger else None
            
            if st_p_path and os.path.exists(st_p_path):
                 
                # --- Top: Image Grid ---
                with st.container(border=True, height=300):
                    s_imgs = sorted(os.listdir(st_p_path))[:20] 
                    if s_imgs:
                        g_cols = st.columns(4)
                        for j, sn in enumerate(s_imgs):
                            with g_cols[j%4]:
                                img_full_p = os.path.join(st_p_path, sn)
                                st.image(img_full_p, use_column_width=True)
                                
                                # Per-image controls (Popover)
                                with st.popover("转移/删除", use_container_width=True):
                                    if st.button("↪️ 转移", key=f"tr_s_{current_stranger}_{sn}", use_container_width=True):
                                        transfer_dialog_func(img_full_p, 'stranger')
                                    
                                    if st.button("🗑️ 删除", key=f"del_img_s_{current_stranger}_{sn}", type="primary", use_container_width=True):
                                        os.remove(img_full_p)
                                        st.rerun()
                    else:
                        st.info("暂无图像")

                # --- Bottom: Person Actions (Single Row) ---
                st.markdown("---")
                
                # Actions: Rename | Convert | Blacklist | Delete
                act_cols = st.columns(4)
                
                with act_cols[0]:
                    if st.button("✏️ 重命名", key=f"btn_ren_s_{current_stranger}", use_container_width=True):
                        rename_dialog_func(current_stranger, 'stranger')

                with act_cols[1]:
                    if st.button("👥 归档为成员", use_container_width=True, type="primary", key=f"btn_valid_{current_stranger}"):
                        target_p = os.path.join(manager.face_bank_path, current_stranger)
                        if not os.path.exists(target_p): 
                            shutil.move(st_p_path, target_p)
                            manager.load_face_bank()
                            st.session_state.member_sel = current_stranger
                            st.session_state.stranger_sel = None
                            st.success(f"已转正: {current_stranger}")
                            st.rerun()
                        else:
                            st.error("同名成员已存在")
            
                with act_cols[2]:
                    if st.button("🚫 拉黑", use_container_width=True, key=f"btn_bl_{current_stranger}"):
                        target_p = os.path.join(manager.face_bank_path, current_stranger)
                        if not os.path.exists(target_p):
                            shutil.move(st_p_path, target_p)
                            
                        if current_stranger not in manager.blacklist:
                            manager.blacklist.append(current_stranger)
                            manager.save_blacklist()
                            
                        manager.load_face_bank()
                        st.session_state.stranger_sel = None
                        st.warning(f"已拉黑: {current_stranger}")
                        st.rerun()

                with act_cols[3]:
                    if st.button("🗑️ 删除", use_container_width=True, key=f"btn_del_s_{current_stranger}"):
                        shutil.rmtree(st_p_path)
                        st.session_state.stranger_sel = None
                        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # --- 3. 黑名单管理 (Blacklist) ---
    st.markdown('<div class="premium-card"><div class="section-header">🚫 黑名单管理</div>', unsafe_allow_html=True)
    
    if not manager.blacklist:
        st.info("当前黑名单为空。")
    else:
        # Layout: Left List, Right Details
        b_c1, b_c2 = st.columns([1, 2.5])
        
        # --- Column 1: Blacklist Selection ---
        with b_c1:
            st.markdown("**🛑 黑名单列表**")
            with st.container(border=True, height=350):
                current_bl = st.session_state.get("blacklist_sel", None)
                if not current_bl and manager.blacklist:
                    current_bl = manager.blacklist[0]
                    st.session_state.blacklist_sel = current_bl
                
                if current_bl not in manager.blacklist:
                     current_bl = manager.blacklist[0] if manager.blacklist else None
                     st.session_state.blacklist_sel = current_bl

                for bl_name in manager.blacklist:
                    is_active = (current_bl == bl_name)
                    if st.button(f"🚫 {bl_name}", key=f"btn_bl_list_{bl_name}", use_container_width=True,
                                 type="primary" if is_active else "secondary"):
                        st.session_state.blacklist_sel = bl_name
                        st.rerun()
        
        # --- Column 2: Details & Actions ---
        with b_c2:
            st.markdown(f"**⚙️ 操作: `{current_bl}`**")
            
            if current_bl:
                bl_p_path = os.path.join(manager.face_bank_path, current_bl)
                exists_on_disk = os.path.exists(bl_p_path)
                
                # Top: Image Preview
                with st.container(border=True, height=300):
                     if exists_on_disk:

                          bl_imgs = [f for f in os.listdir(bl_p_path) if f.lower().endswith(('.jpg', '.png'))]
                          if bl_imgs:
                               g_cols = st.columns(4)
                               for j, sn in enumerate(bl_imgs):
                                   with g_cols[j%4]:
                                       img_full_p = os.path.join(bl_p_path, sn)
                                       st.image(img_full_p, use_column_width=True)
                                       
                                       # Per-image controls (Popover)
                                       with st.popover("转移/删除", use_container_width=True):
                                           if st.button("↪️ 转移", key=f"tr_bl_{current_bl}_{sn}", use_container_width=True):
                                               transfer_dialog_func(img_full_p, 'blacklist')
                                           
                                           if st.button("🗑️ 删除", key=f"del_img_bl_{current_bl}_{sn}", type="primary", use_container_width=True):
                                               os.remove(img_full_p)
                                               st.rerun()
                          else:
                               st.info("该目录无图像样本")
                     else:
                          st.warning("⚠️ 磁盘上未找到对应的文件夹 (仅在名单中)")

                # Bottom: Actions (Single Row)
                st.markdown("---")
                
                act_cols = st.columns(3)
                with act_cols[0]:
                    if st.button("✏️ 重命名", key=f"btn_ren_bl_{current_bl}", use_container_width=True):
                        rename_dialog_func(current_bl, 'blacklist')
                
                with act_cols[1]:
                    if st.button("✅ 恢复成员", use_container_width=True, type="primary", key=f"btn_restore_{current_bl}"):
                        if current_bl in manager.blacklist:
                            manager.blacklist.remove(current_bl)
                            manager.save_blacklist()
                        manager.load_face_bank()
                        st.session_state.member_sel = current_bl
                        st.session_state.blacklist_sel = None 
                        st.success(f"已恢复")
                        st.rerun()
                
                with act_cols[2]:
                    if st.button("🚨 彻底删除", use_container_width=True, key=f"btn_del_bl_{current_bl}"):
                        if current_bl in manager.blacklist:
                            manager.blacklist.remove(current_bl)
                            manager.save_blacklist()
                        if exists_on_disk:
                            shutil.rmtree(bl_p_path)
                        manager.load_face_bank()
                        st.session_state.blacklist_sel = None
                        st.warning("已删除")
                        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# --- Main App Logic ---
def main():
    # --- Navigation ---

    with st.sidebar:
        selected_module = option_menu(
            "智能视界", 
            ["图像的本质", "卷积实验室", "神经网络实验室", "YOLO 实验室", "SAM 实验室", "人脸实验室"],
            icons=['palette', 'filter', 'cpu', 'eye', 'layers', 'person-bounding-box'],
            menu_icon="cast", 
            default_index=5,
            styles={"nav-link-selected": {"background-color": "#663399"}}
        )

    st.sidebar.divider()
    st.sidebar.caption("Tsinghua University\nGeneral Education Course 2026")

    # Routing
    if selected_module == "图像的本质":
        render_color_space_lab(DATA_PATH)
    elif selected_module == "卷积实验室":
        render_convolution_lab(DATA_PATH)
    elif selected_module == "神经网络实验室":
        render_nn_lab()
    elif selected_module == "YOLO 实验室":
        render_yolo_tab()
    elif selected_module == "SAM 实验室":
        render_sam_lab()
    elif selected_module == "人脸实验室":
        render_face_lab()

if __name__ == "__main__":
    main()

