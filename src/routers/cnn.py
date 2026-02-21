import asyncio
import io
import base64
import json
import numpy as np
from PIL import Image
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from src.network import SimpleCNN, load_mnist_data, get_random_sample, calculate_accuracy, tensor_to_img_array
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

router = APIRouter()

# Global variables to handle background training properly if needed, 
# although a per-websocket session model is cleaner.
# To keep memory bounded, we will instantiate model per websocket.

def encode_pil_base64(pil_img: Image.Image, format="PNG") -> str:
    buffered = io.BytesIO()
    if pil_img.mode in ["HSV", "YCbCr"]:
        pil_img = pil_img.convert("RGB")
    pil_img.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def generate_feature_map_b64(model, input_tensor):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        feats = model.get_features(input_tensor.to(device).float())
        
    num_layers = getattr(model, 'num_layers', 2)
    viz_names = []
    for idx in range(1, num_layers + 1):
        if f'Conv{idx}' in feats: viz_names.append(f'Conv{idx}')
        if f'Pool{idx}' in feats: viz_names.append(f'Pool{idx}')
        
    layer_imgs = []
    for layer_name in viz_names:
        f = feats[layer_name][0]
        # Take first 8 channels
        sub_imgs = [tensor_to_img_array(f[k]) for k in range(min(8, f.shape[0]))]
        row = np.hstack([np.array(Image.fromarray(im).resize((40, 40), resample=Image.NEAREST)) for im in sub_imgs])
        layer_imgs.append(row)
        
    if layer_imgs:
        max_w = max(row.shape[1] for row in layer_imgs)
        fixed_rows = []
        for row in layer_imgs:
            if row.shape[1] < max_w:
                row = np.pad(row, ((0,0), (0, max_w - row.shape[1])), 'constant')
            fixed_rows.append(row)
        
        # Stack rows with 5px spacer
        full = fixed_rows[0]
        for row in fixed_rows[1:]:
            full = np.vstack([full, np.zeros((5, max_w), dtype=np.uint8), row])
        
        return encode_pil_base64(Image.fromarray(full))
    return None

@router.get("/status")
async def check_status():
    return {"has_torch": HAS_TORCH}

@router.websocket("/ws/train")
async def websocket_train(websocket: WebSocket):
    await websocket.accept()
    if not HAS_TORCH:
        await websocket.send_json({"type": "error", "msg": "PyTorch is not installed on the server."})
        await websocket.close()
        return

    # Session state
    model = None
    optimizer = None
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")
    is_training = False
    
    # Keep track of task
    train_task = None
    stop_event = asyncio.Event()
    pause_event = asyncio.Event()
    pause_event.set() # Not paused initially
    
    train_data, test_data = None, None

    async def training_loop(config):
        global _global_testing_model
        nonlocal model, optimizer, train_data, test_data
        try:
            lr = float(config.get("lr", 0.01))
            num_epochs = int(config.get("epochs", 3))
            batch_size = int(config.get("batch_size", 64))
            layer_configs = config.get("layers", [16, 32])
            
            # Init Model (Force float32 for compatibility)
            model = SimpleCNN(layer_configs=layer_configs)
            model.to(device).float()
            # Optimizer Selection
            opt_name = config.get("optimizer", "Adam")
            if opt_name == "SGD":
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            else:
                optimizer = optim.Adam(model.parameters(), lr=lr)
            
            # Loss Function Selection
            loss_name = config.get("loss_fn", "CrossEntropy")
            if loss_name == "NLLLoss":
                criterion = nn.NLLLoss()
                use_nll = True
            else:
                criterion = nn.CrossEntropyLoss()
                use_nll = False
            
            # Load Data
            await websocket.send_json({"type": "status", "msg": "Loading MNIST dataset..."})
            
            # Run blocking load in executor
            loop = asyncio.get_event_loop()
            train_data, test_data = await loop.run_in_executor(None, load_mnist_data)
            
            if train_data is None:
                await websocket.send_json({"type": "error", "msg": "Failed to load MNIST data."})
                return
                
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            
            show_visuals = config.get("show_visuals", True)
            await websocket.send_json({"type": "status", "msg": "Starting training..."})
            
            # Initialize visualization sample
            viz_input_tensor, viz_label = get_random_sample(train_data)
            
            global_step = 0
            for epoch in range(num_epochs):
                if stop_event.is_set(): break
                
                for i, (images, labels) in enumerate(train_loader):
                    if stop_event.is_set(): break
                    await pause_event.wait()
                    
                    images, labels = images.to(device).float(), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    
                    # If NLLLoss is used, we need log_softmax since SimpleCNN doesn't have it
                    if use_nll:
                        loss = criterion(torch.nn.functional.log_softmax(outputs, dim=1), labels)
                    else:
                        loss = criterion(outputs, labels)
                    
                    loss.backward()
                    optimizer.step()
                    
                    # Rule: Update loss status every 10 steps
                    if global_step % 10 == 0:
                        msg = f"Training... Epoch [{epoch+1}/{num_epochs}], Step [{global_step}]"
                        await websocket.send_json({
                            "type": "progress",
                            "epoch": epoch,
                            "step": global_step,
                            "loss": loss.item(),
                            "msg": msg
                        })
                    
                    # Rule 2: Change random sample every 200 global steps
                    if show_visuals and global_step % 200 == 0 and global_step > 0:
                        viz_input_tensor, viz_label = get_random_sample(train_data)

                    # Rule 3 & 4 & 5: Update feature maps and probs every 10 steps if enabled
                    if show_visuals and global_step % 10 == 0:
                        model.eval()
                        with torch.no_grad():
                            viz_input_float = viz_input_tensor.to(device).float()
                            feats = model.get_features(viz_input_float)
                            out_viz = model(viz_input_float)
                            probs = torch.nn.functional.softmax(out_viz, dim=1)[0].cpu().numpy()
                        model.train()

                        # Create composite feature map image via shared helper
                        feat_img_b64 = generate_feature_map_b64(model, viz_input_tensor)
                        
                        if feat_img_b64:
                            await websocket.send_json({
                                "type": "visuals",
                                "feat_img": feat_img_b64,
                                "true_label": int(viz_label),
                                "probs": probs.tolist()
                            })
                    
                    global_step += 1
                    # Yield control
                    await asyncio.sleep(0.0001)
                    
                    # Yield control to event loop to allow receiving messages
                    await asyncio.sleep(0.001)

            if not stop_event.is_set():
                await websocket.send_json({"type": "status", "msg": "Calculating accuracy..."})
                
                # Accuracy calc block in executor
                def calc_accs():
                    t_loader = DataLoader(train_data, batch_size=1000, shuffle=False)
                    v_loader = DataLoader(test_data, batch_size=1000, shuffle=False)
                    tr_a = calculate_accuracy(model, t_loader, device)
                    te_a = calculate_accuracy(model, v_loader, device)
                    return tr_a, te_a
                
                train_acc, test_acc = await loop.run_in_executor(None, calc_accs)
                
                # Sync to global model for Test Tab
                _global_testing_model = model.cpu() # Store the trained state
                
                await websocket.send_json({
                    "type": "finished",
                    "train_acc": train_acc,
                    "test_acc": test_acc
                })
            else:
                # Even if stopped, sync what we have
                _global_testing_model = model.cpu()
                await websocket.send_json({"type": "stopped"})

        except asyncio.CancelledError:
            # Sync partial results before exit
            if model:
                _global_testing_model = model.cpu()
            await websocket.send_json({"type": "stopped"})
        except Exception as e:
            import traceback
            traceback.print_exc()
            await websocket.send_json({"type": "error", "msg": str(e)})

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            action = payload.get("action")
            
            if action == "start":
                if train_task and not train_task.done():
                    train_task.cancel()
                    try: await train_task
                    except: pass
                stop_event.clear()
                pause_event.set()
                config = payload.get("config", {})
                train_task = asyncio.create_task(training_loop(config))
                
            elif action == "stop":
                if train_task and not train_task.done():
                    train_task.cancel()
                stop_event.set()
                
            elif action == "pause":
                pause_event.clear()
                
            elif action == "resume":
                pause_event.set()
                
    except WebSocketDisconnect:
        if train_task and not train_task.done():
            train_task.cancel()
    except Exception as e:
        print(f"WS error: {e}")
        if train_task and not train_task.done():
            train_task.cancel()

# We need a shared model state between the WS trainer and the generic Test endpoint,
# but for simplicity of this translation architecture, we will instantiate a pre-trained
# or just random model if tested before training. In a real app, model registry is needed.
# For now, we will offer a simple test endpoint that creates a dummy model or uses a globally cached one.

_global_testing_model = None

@router.post("/test/random")
async def test_random():
    global _global_testing_model
    if not HAS_TORCH:
         raise HTTPException(status_code=500, detail="PyTorch not installed")
    
    loop = asyncio.get_event_loop()
    # load_mnist_data and model inference are synchronous
    def run_sync_test():
        global _global_testing_model
        train_d, test_d = load_mnist_data()
        if not test_d: return None
            
        if _global_testing_model is None:
            _global_testing_model = SimpleCNN().float()
            
        input_t, label = get_random_sample(test_d)
        input_t = input_t.float() 
        inv_t = input_t[0] * 0.3081 + 0.1307
        inv_t = torch.clamp(inv_t, 0, 1)
        disp_img = transforms.ToPILImage()(inv_t)
        
        _global_testing_model.eval()
        with torch.no_grad():
            output = _global_testing_model(input_t)
            _, pred = torch.max(output, 1)
            p_list = torch.nn.functional.softmax(output, dim=1)[0].numpy().tolist()
            
        feat_img_b64 = generate_feature_map_b64(_global_testing_model, input_t)
        
        return {
            "image_b64": encode_pil_base64(disp_img),
            "true_label": int(label),
            "prediction": int(pred.item()),
            "probs": p_list,
            "feat_img": feat_img_b64
        }

    res = await loop.run_in_executor(None, run_sync_test)
    if res is None:
        raise HTTPException(status_code=500, detail="Failed to load data")
    return res

@router.post("/test/upload")
async def test_upload(file: UploadFile = File(...)):
    global _global_testing_model
    if not HAS_TORCH:
         raise HTTPException(status_code=500, detail="PyTorch not installed")
         
    contents = await file.read()
    loop = asyncio.get_event_loop()
    
    def run_sync_upload():
        global _global_testing_model
        pil_img = Image.open(io.BytesIO(contents)).convert('L').resize((28, 28))
        
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        input_t = transform(pil_img).unsqueeze(0).float()
        
        if _global_testing_model is None:
            _global_testing_model = SimpleCNN().float()
            
        _global_testing_model.eval()
        with torch.no_grad():
            output = _global_testing_model(input_t)
            _, pred = torch.max(output, 1)
            p_list = torch.nn.functional.softmax(output, dim=1)[0].numpy().tolist()
            
        feat_img_b64 = generate_feature_map_b64(_global_testing_model, input_t)
            
        return {
            "image_b64": encode_pil_base64(pil_img),
            "prediction": int(pred.item()),
            "probs": p_list,
            "feat_img": feat_img_b64
        }

    return await loop.run_in_executor(None, run_sync_upload)
