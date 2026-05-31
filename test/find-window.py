import pygetwindow as gw


def list_window_titles():
    print("正在扫描所有打开的窗口...\n")
    # 获取当前所有窗口对象
    all_windows = gw.getAllWindows()

    # 提取标题并过滤掉空标题
    titles = [win.title for win in all_windows if win.title.strip() != ""]

    # 排序输出，方便查看
    for i, title in enumerate(sorted(titles), 1):
        print(f"窗口 {i}: {title}")

    print("\n--- 扫描完成 ---")
    print("请在上方列表中找到你的游戏窗口名称，并复制到 GAME_WINDOW_TITLE 中。")


if __name__ == "__main__":
    list_window_titles()