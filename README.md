# 前言
* 本项目参考了这个库[MapleStoryAutoLevelUp](https://github.com/KenYu910645/MapleStoryAutoLevelUp) 
* 只要是市面上的079版本的冒险岛都可以使用，测试过的版本有：wingms, mapleStoryGo。
* 还有，开挂是不对的，本人已经被封了3个号。

# 功能
 * [x] mini地图识别 (还有问题，需要调试)
 * [x] 角色识别 
 * [x] 怪物识别
 * [ ] 地图绘制 
 * [x] 自动打怪
 * [x] 自动吃药
 * [x] 测谎报警 (需要看看国服什么样子的策划界面)


# 1、识别角色 + 怪物 + miniMap
## 1-1、演示
![识别怪物演示](demo_gif/recognize_user_map_monster.gif)
## 1-2、素材库找到怪物素材
* [素材下载地址](https://maplestory.wiki/GMS/83/mob/1110101)
* 把走动的素材下载下载，放到`monster/下载的怪物名称`文件夹下
* 大概长这样![下载的怪物素材demo](demo_gif/monster.png)
* 如何调整配置
  * 打开`config/config_data.yaml`，找到`mob`字段
  * 添加新的怪物配置，示例如下：只要改这三个字段就行
```yaml
monsters: ["black_axe_stump"]  # 指定怪物文件夹，数组可多个
diff_thres: 0.46               # 匹配阈值（越小越严格）
scales: [1]                    # 缩放比例，官方素材不匹配时改: [0.7, 0.85, 1.0, 1.15]            # 每N帧检测一次 (1=每帧, 3=流畅优先)
  ```

# 2、地图绘制
TODO

