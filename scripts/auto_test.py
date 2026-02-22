import requests

# 1. 之前用户成功生成 session 时的已知可用 mock 参数
# 实际中，后端是通过全局 dictionary 缓存 session 的
# 我们可以传一个假的 UUID 取现有的或者就直接取个固定的看后端逻辑
SESSION_ID = "test-session"
OBJ_ID = 1
FRAME_IDX = 0

print("==> 1. Start Session")
# 为了防止刚才重启或者因为错误清理了状态，我们需要重新发一份 start_session 来挂载模型追踪状态
res = requests.post("http://127.0.0.1:8000/api/sam/video/start_session", json={"session_id": SESSION_ID, "video_path": "video/sam_demo1.mp4"})
print(res.status_code, res.text)


if res.status_code == 200:
    SESSION_ID = res.json().get("session_id", SESSION_ID)

print("\n==> 2. Add Prompt (Interactive Point on Frame 0)")
# 模拟用户在图像上点击 (坐标归一化/非归一化视前端原逻辑而定，先给一个绝对坐标)
prompt_data = {
    "session_id": SESSION_ID,
    "frame_idx": FRAME_IDX,
    "obj_id": OBJ_ID,
    "points": [{"x": 150, "y": 150, "label": 1}]
}

res2 = requests.post("http://127.0.0.1:8000/api/sam/video/add_prompt", json=prompt_data)
if res2.status_code == 200:
    print("SUCCESS! Mask extracted:", len(res2.json().get('masks', [])))
else:
    print(f"FAILED with {res2.status_code}:", res2.text)
