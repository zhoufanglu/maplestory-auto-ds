# 前言
* 本项目参考了这个库 [MapleStoryAutoLevelUp](https://github.com/KenYu910645/MapleStoryAutoLevelUp)
* 适用于市面上的 079/083 版本冒险岛（wingms、mapleStoryGo、国服怀旧服等）
* 仅记录和分享

## 演示视频

完整视频：
<video src="video_demo/2026-10-18%2020-00-04.mp4" controls width="800"></video>

GIF 动图预览（前 30 秒）：
![演示 GIF](video_demo/demo.gif)

---

# 1、角色识别（角色定位）

## 文件目录

| 文件 | 作用 |
|------|------|
| `character/fashi.png` | 角色**外观**模板（主定位，蓝色角色） |
| `character/title.png` | 角色**称号**模板（兜底定位） |

## 配置说明（`config/config_custom.yaml`）

优先级：**外观为主，称号兜底**。

```yaml
appearance:                    # 外观定位（fashi.png）
  enable: true
  template: "character/fashi.png"
  match_threshold: 0.85        # CCORR 匹配分，>=0.85 用外观，<0.85 用称号

title:                         # 称号定位（title.png）
  enable: true
  mode: "grayscale"            # grayscale / white_mask
  match_threshold: 0.28        # SQDIFF 分数，<0.28 算命中（越小越像）
```

- 右下角会显示 `AP: x.xx [v]`（外观分）和 `Title: x.xx [x]`（称号分），`[v]` 表示当前用哪个定位。
- 外观匹配分掉到 0.85 以下（比如攻击动画）就自动切到称号定位。

---

# 2、怪物识别

## 文件目录

怪物素材放在 `monster/` 下，**每个怪物一个文件夹**：

```
monster/
  zhang_yu/
    move_0.png
    move_1.png
    ...
  lan_mo_gu/
    stand_0.png
    move_0.png
    ...
```

- 素材下载地址：[https://maplestory.wiki/GMS/83/mob/1110101](https://maplestory.wiki/GMS/83/mob/1110101)
- 把怪物「走动/站立」的图片放进对应文件夹即可。

## 配置说明（`config/config_custom.yaml`）

```yaml
monster_detect:
  mode: "color"                # color / grayscale / contour_only
  diff_thres: 0.25             # 全局默认匹配阈值（越小越严格）
  scales: [1.1]                # 全局默认缩放比例
  min_color_spread: 30         # 颜色纯度下限（过滤云朵等误识别）

  # 每个怪物单独配置（key = 文件夹名）
  monsters:
    zhang_yu:
      diff_thres: 0.302
      scales: [1.1, 1.2]
    lan_mo_gu:
      diff_thres: 0.20
      scales: [1.1]
      min_color_spread: 30     # 暗色怪物可调低，如 0
```

- 每个怪物的 `diff_thres` / `scales` / `min_color_spread` 会覆盖全局默认。
- 怪物识别框颜色：**棕红色 `(42,42,165)`**，框上显示 `怪物名:相似度`。
- 顶部中间显示 `Monsters: N`（识别到的怪物数量）。

---

# 3、地图圈选 + 地图绘制（路线录制）

## 3-1、地图圈选（小地图框选）

在监视器（NewMaple）窗口里：

1. 按 **`m`** 键 → 进入圈选模式（右上角出现 3 倍放大镜，方便对准边框）。
2. 按住左键**拖拽框住小地图**，松手即选定。
3. 控制台会打印 `[Minimap] → 写进 config: x=... y=... width=... height=...`，把这 4 个值填进 `config_custom.yaml` 的 `minimap` 段。

```yaml
minimap:
  fixed: true                  # true=用下面固定坐标
  x: 23
  y: 130
  width: 122
  height: 106
  player_color: [136, 255, 255]  # 角色小黄点 BGR (#ffff88)
  player_tolerance: 8
  width_boundary_pct: 0.6      # 角色只能在中间 30%~70% 内移动（0=关闭）
```

- 小地图用**红色框**标记，角色点用**绿色点 P** 标记，移动边界用**黄色虚线**标记。

## 3-2、地图绘制（录制路线）

```bash
.venv/Scripts/python.exe tools/routeRecorder.py --new_map duoluozhuyi --cfg custom
```

参数：
- `--new_map`：地图名（输出到 `minimaps/<地图名>/`）
- `--cfg`：用哪个配置（`custom` = `config_custom.yaml`）

**继续录制已有地图（不覆盖）**：
```bash
.venv/Scripts/python.exe tools/routeRecorder.py --new_map duoluozhuyi --cfg custom --map minimaps/duoluozhuyi/map.png
```

录制按键（窗口里）：
- **F1**：开始/暂停录制（默认关，先走到起点再按）
- **F3**：到终点放黄点 + 保存 `route*.png`（录多段就每段终点按一次 F3）
- **Q**：退出

录制结果：`minimaps/<地图名>/` 下生成 `map.png`（拼接大地图）+ `route1.png`、`route2.png`…（路线），`main.py` 会按 route1 → route2 → … 循环走。

---

# 4、告警（测谎报警）

## 开关

在 `config/config_custom.yaml`：

```yaml
bot:
  warning: true               # true=开 false=关
```

## 文件地址

| 文件 | 作用 |
|------|------|
| `warning/warning.png` | 告警模板截图（检测时用来匹配弹窗） |
| `warning/warning.mp3` | 报警音频 |
| `warning/warning.py` | 告警检测逻辑 |

- 检测到测谎弹窗会**播放报警音**并在控制台打印 `[Warning] 检测到测谎报警弹窗！`。
- 匹配阈值在 `warning/warning.py` 里：`CONFIDENCE_LEVEL = 0.5`（0~1，越大越严格）。
- 如果报警图变了，替换 `warning/warning.png` 即可。

## 手动截图

监视器窗口按 **`s`** 键截图，保存到 `screenshot/` 目录（文件名带时间戳）。
