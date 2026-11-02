# 冒险岛测谎小游戏（Lie Detector）分析

> 本目录属于 `lie-detector-test` 分支，用于分析并尝试自动通过《冒险岛》怀旧服中的测谎小游戏。

---

## 1. 视频来源

| 项目 | 内容 |
|------|------|
| 文件 | `assets/2026-09-29_16-52-17_lie_detector.mp4` |
| 分辨率 | 1366 × 796 |
| 帧率 | 10 fps |
| 时长 | 20 秒 |
| 提取帧 | 见 `frames/` 目录 |

---

## 2. 游戏规则

测谎小游戏是《冒险岛》的反挂机验证机制。

1. 弹窗标题为 **"谎言探测仪"**。
2. 倒计时 3 秒后进入小游戏。
3. 游戏中会显示一个**逐渐变透明的白色/浅色图形**，图形内部有一个**绿色圆圈目标点**。
4. 玩家需要用鼠标光标跟随目标点移动。
5. 系统根据鼠标光标位于图形上的时间判定通过或失败。

---

## 3. 关键观察

| 时间点 | 画面内容 |
|--------|----------|
| 0 秒 | 规则说明框：提示 3 秒后开始，需移动鼠标跟随图形 |
| 5 秒 | 显示"开始"，出现白色岩石背景和目标图形 |
| 10 秒 | 目标图形移动到屏幕偏下位置，绿色目标点清晰可见 |
| 15 秒 | 目标图形移动到屏幕偏上位置，透明度降低 |

---

## 4. 自动化的技术思路

### 4.1 检测目标点

最稳定的视觉特征：

- **绿色圆圈目标点**：颜色鲜明，在岩石背景中对比度高。
- 使用 HSV 颜色空间检测绿色区域。
- 过滤小噪点，选择面积最大的绿色圆斑作为目标。

### 4.2 跟随策略

- 计算目标点相对于屏幕中央的偏移。
- 使用 PyAutoGUI / 其他鼠标控制库移动鼠标。
- 为保持平滑，采用比例控制（PID 可选）。

### 4.3 状态机

```
[等待弹窗] -> [检测到"谎言探测仪"标题] -> [等待"开始"]
      -> [开始跟随绿色目标点] -> [检测到"通过"/弹窗消失 -> 结束]
```

---

## 5. 文件说明

| 文件 | 作用 |
|------|------|
| `README.md` | 本说明文档 |
| `assets/*.mp4` | 原始测谎视频 |
| `frames/*.jpg` | 视频关键帧截图 |
| `detector_demo.py` | 绿色目标点检测 demo |
| `mouse_follower.py` | 鼠标跟随 demo |
| `output/` | 生成的调试图像（已加入 .gitignore） |

---

## 6. 环境安装

当前环境：macOS + Python 3.9

```bash
# 升级 pip（可选）
python3 -m pip install --upgrade pip

# 安装依赖
python3 -m pip install opencv-python numpy pyautogui
```

已安装版本：

| 包 | 版本 |
|----|------|
| opencv-python | 5.0.0.93 |
| numpy | 2.0.2 |
| pyautogui | 0.9.54 |

---

## 7. 运行方式

### 7.1 目标检测 Demo

```bash
# 单张图片测试（无 GUI，保存结果图）
python detector_demo.py --image frames/frame_10s.jpg --headless

# 视频测试（无 GUI，逐帧输出目标坐标）
python detector_demo.py --video "assets/2026-09-29_16-52-17_lie_detector.mp4" --headless
```

### 7.2 鼠标跟随 Demo

```bash
# 可视化模式（需要 GUI）
python mouse_follower.py --video "assets/2026-09-29_16-52-17_lie_detector.mp4"

# 无 GUI 模式，每 10 帧保存一张调试图像
python mouse_follower.py --video "assets/2026-09-29_16-52-17_lie_detector.mp4" --headless

# 实际移动鼠标（谨慎使用）
python mouse_follower.py --video "assets/2026-09-29_16-52-17_lie_detector.mp4" --move-mouse
```

### 7.3 测试结果

对视频 `2026-09-29_16-52-17_lie_detector.mp4` 的处理结果：

- 总帧数：200 帧
- 分辨率：1366 × 796
- 帧率：10 fps
- 检测结果：**成功在每一帧中定位绿色目标点**
- 输出图像：`output/frame_10s_result.jpg` 等

示例检测坐标：

| 帧 | 目标坐标 |
|----|----------|
| frame_5s.jpg | (660, 391) |
| frame_10s.jpg | (734, 458) |
| frame_15s.jpg | (742, 271) |

---

## 8. 实现细节

### 8.1 颜色检测

使用 HSV 颜色空间提取绿色目标点：

```python
lower_green = np.array([35, 80, 80])
upper_green = np.array([85, 255, 255])
```

### 8.2 ROI 限制

为避免 UI 按钮、背景植物等干扰，只在测谎小游戏窗口区域内检测：

```python
roi = (330, 110, 640, 520)
```

### 8.3 鼠标跟随平滑

使用简单的指数平滑：

```python
x = int(last_x + (target_x - last_x) * smooth_factor)
y = int(last_y + (target_y - last_y) * smooth_factor)
```

---

## 9. 注意事项

- 当前测试环境为 macOS，但代码尽量保持跨平台。
- 实际游戏运行在 Windows 上，最终部署需要在 Windows 环境测试。
- 本项目仅供学习研究，请勿用于违反游戏用户协议的场景。
