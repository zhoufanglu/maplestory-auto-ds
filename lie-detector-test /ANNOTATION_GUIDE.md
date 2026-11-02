# YOLOv8 标注指南（LabelImg）

## 1. 安装 LabelImg

```bash
pip install labelImg
```

## 2. 启动 LabelImg

```bash
cd "/Users/lufangzhou/Documents/work-space/personal/maplestory-auto-ds/lie-detector-test "
labelImg dataset/images/train dataset/classes.txt
```

> 注意：`classes.txt` 已经创建好，内容为 `target`。

## 3. 设置 LabelImg

打开 LabelImg 后，确保：

1. **Open Dir**: `lie-detector-test/dataset/images/train`
2. **Change Save Dir**: `lie-detector-test/dataset/labels/train`
3. 左侧勾选 **Auto Save mode**
4. 格式选择 **YOLO**（不是 PascalVOC）

## 4. 标注规则

### 标什么？

标出**白色/浅色目标图形**（玩家需要把鼠标放上去的形状）。

不要标：
- 绿色圆点（鼠标）
- 岩石背景
- 规则说明页底部的星形示意图

### 怎么框？

用矩形框紧贴目标图形的外接矩形：

```
┌─────────────────────┐
│                     │
│    ┌──────────┐     │
│    │  白色    │     │
│    │  图形    │     │
│    └──────────┘     │
│                     │
└─────────────────────┘
```

### 标签名

统一使用：`target`

按 `w` 画框，在弹出的对话框里输入 `target`。

## 5. 标注对象示例

| 情况 | 是否标注 | 说明 |
|------|----------|------|
| 游戏开始后的白色不规则图形 | ✅ 标注 | 这就是目标 |
| 规则说明页 | ❌ 不标 | 没有游戏目标 |
| 游戏结束后的背景 | ❌ 不标 | 没有目标 |
| 白色图形部分被遮挡 | ✅ 标注可见部分 | 框住可见区域 |

## 6. 验证集

验证集图片在 `dataset/images/val/`，也需要标注，保存到 `dataset/labels/val/`。

## 7. 检查标注格式

标注完成后，每个 `.txt` 文件应该长这样：

```
0 0.543 0.478 0.123 0.234
```

含义：
- `0`: 类别编号（target）
- `0.543`: 目标中心 x / 图片宽度
- `0.478`: 目标中心 y / 图片高度
- `0.123`: 目标宽度 / 图片宽度
- `0.234`: 目标高度 / 图片高度

## 8. 标完之后的下一步

1. 把图片和对应的 `.txt` 文件保持同名
2. 运行训练脚本：

```bash
python lie-detector-test/train_yolo.py
```

## 9. 快捷键

| 快捷键 | 功能 |
|--------|------|
| `w` | 画框 |
| `d` | 下一张 |
| `a` | 上一张 |
| `Ctrl+s` | 保存 |
| `Ctrl+z` | 撤销 |
