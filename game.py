import cv2
import pygetwindow as gw
import pyautogui
import numpy as np
import time
from PIL import Image
from cnocr import CnOcr
class GameAutomator:
    def __init__(self, window_title):
        self.window = gw.getWindowsWithTitle(window_title)[0]
        self.ocr = CnOcr(det_model_name='naive_det', rec_model_name='ch_PP-OCRv3', cand_alphabet="123456789")
        self.cell_width = 0
        self.cell_height = 0
        self.result = []
        self.used = []
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None

    def capture_window(self):
        """Capture the entire game window"""
        self.window.activate()
        left, top, right, bottom = self.window.left, self.window.top, self.window.right, self.window.bottom
        screenshot = pyautogui.screenshot(region=(left, top, right-left, bottom-top))
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    def process_image(self, frame):
        """Process the captured frame for OCR"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        cropped = frame[y:y+h, x:x+w]
        
        pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        pil_img = pil_img.convert('L').point(lambda p: 0 if p < 25 else 255)
        opencv_img = np.array(pil_img)
        kernel = np.ones((3, 3), np.uint8)
        close_image = cv2.morphologyEx(opencv_img, cv2.MORPH_ELLIPSE, kernel)
        cv2.imwrite("preprocessed_num6.jpg", close_image)
        return close_image

    def recognize_numbers(self, processed_img):
        """Recognize the numbers from the processed image"""
        res = self.ocr.ocr(processed_img)
        text = ''.join([line['text'] for line in res if line['text'].strip()])
        
        self.result = []
        for i in range(16):
            row = []
            for j in range(10):
                index = i * 10 + j
                row.append(int(text[index]) if index < len(text) else 0)
            self.result.append(row)
        self.used = [[False]*10 for _ in range(16)]
        
        # Calculate cell dimensions
        h, w = processed_img.shape
        self.cell_width = w // 10
        self.cell_height = h // 16


    def check_rectangle(self, height, width):
        """检测矩形区域是否能组成10，并标记被选中的数字"""
        for start_row in range(16 - height + 1):  # 确保矩形在网格内
            for start_col in range(10 - width + 1):  # 确保矩形在网格内
                total = 0
                # 检查是否所有格子都未被使用
                if True:#all(not self.used[start_row + i][start_col + j] for i in range(height) for j in range(width)):
                    # 计算矩形区域内所有数字的和
                    for i in range(height):
                        for j in range(width):
                            total += self.result[start_row + i][start_col + j]
                    if total == 10:  # 如果和为10
                        print(f"矩形区域：起始位置 ({start_row}, {start_col})，大小 {height}x{width}，组成10")
                        # 标记该矩形区域内的所有格子为已使用
                        for i in range(height):
                            for j in range(width):
                                self.used[start_row + i][start_col + j] = True
                                self.result[start_row + i][start_col + j] = 0
                        # 执行滑动操作
                        self.slide_between(start_row, start_col, start_row + height - 1, start_col + width - 1)
                        # # 更新图像
                        # self.draw_numbers()
                        # cv2.imshow("Number Table", self.image)
                        # cv2.waitKey(500)  # 显示更新后的表格
                        return True  # 找到一个符合条件的矩形后退出
        return False  # 没有找到符合条件的矩形
    def check_row(self):
        """检查同一行中多个连续格子的和是否能组成10，并标记已选中的格子"""
        for row in range(16):
            start_col = 0
            while start_col < 10:
                total = 0
                for col in range(start_col, 10):
                    total += self.result[row][col]
                    if total == 10:
                        print(f"第 {row} 行，格子 {start_col} 到 {col} 组成10")
                        for c in range(start_col, col + 1):
                            self.used[row][c] = True  # 标记为已使用
                            self.result[row][c] = 0  # 标记为已使用
                        self.slide_between(row, start_col, row, col)
                        # self.draw_numbers()
                        # cv2.imshow("Number Table", self.image)
                        # cv2.waitKey(500)  # 显示更新后的表格
                        break
                    elif total > 10:
                        break
                start_col += 1
        return False

    def check_col(self):
        """检查同一列中多个连续格子的和是否能组成10，并标记已选中的格子"""
        for col in range(10):
            start_row = 0
            while start_row < 16:
                total = 0
                for row in range(start_row, 16):
                    total += self.result[row][col]
                    if total == 10:
                        print(f"第 {col} 列，格子 {start_row} 到 {row} 组成10")
                        for r in range(start_row, row + 1):
                            self.used[r][col] = True  # 标记为已使用
                            self.result[r][col] = 0  # 标记为已使用
                        self.slide_between(start_row, col, row, col)
                        # self.draw_numbers()
                        # cv2.imshow("Number Table", self.image)
                        # cv2.waitKey(500)  # 显示更新后的表格
                        break
                    elif total > 10:
                        break
                start_row += 1
        return False


    def calibrate_grid(self):
        """精准标定游戏区域"""
        print("请点击游戏窗口左上角第一个方块的左上角（建议放大游戏界面操作）")
        input("按回车开始标定左上角...")
        self.start_x, self.start_y = pyautogui.position()
        
        print("请点击游戏窗口右下角最后一个方块的右下角")
        input("按回车开始标定右下角...")
        end_x, end_y = pyautogui.position()

        # 计算单元格尺寸
        self.cell_w = (end_x - self.start_x) // 10  # 总共有10列
        self.cell_h = (end_y - self.start_y) // 16  # 总共有16行
        print(f"标定结果：起点({self.start_x},{self.start_y}) 单元格{self.cell_w}x{self.cell_h}")

    def get_cell_center(self, row, col):
        """获取指定单元格中心坐标（添加随机偏移防检测）"""
        x = self.start_x + col*self.cell_w + self.cell_w//2 + np.random.randint(-2,3)
        y = self.start_y + row*self.cell_h + self.cell_h//2 + np.random.randint(-2,3)
        return x, y

    def slide_between(self, start_row, start_col, end_row, end_col):
        """精准滑动操作"""
        # 获取实际屏幕坐标
        start_x, start_y = self.get_cell_center(start_row, start_col)
        end_x, end_y = self.get_cell_center(end_row, end_col)
        
        # 模拟人类操作轨迹
        pyautogui.moveTo(start_x, start_y, duration=0.1)
        pyautogui.dragTo(end_x, end_y, duration=0.2, button='left')
        
        # 更新使用状态
        self.used[start_row][start_col] = True
        self.used[end_row][end_col] = True
        time.sleep(0.05)  # 等待动画完成

    def draw_numbers(self):
        """实时绘制数字表格并高亮已消除项"""
        rows, cols = 16, 10
        image = np.ones((600, 400, 3), dtype=np.uint8) * 255
        
        # 绘制网格线
        cell_width = 400 // cols
        cell_height = 600 // rows
        for i in range(rows + 1):
            cv2.line(image, (0, i * cell_height), (400, i * cell_height), (0, 0, 0), 1)
        for j in range(cols + 1):
            cv2.line(image, (j * cell_width, 0), (j * cell_width, 600), (0, 0, 0), 1)

        # 绘制数字
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(rows):
            for j in range(cols):
                text = str(self.result[i][j])
                text_size = cv2.getTextSize(text, font, 0.5, 1)[0]
                x = j * cell_width + (cell_width - text_size[0]) // 2
                y = i * cell_height + (cell_height + text_size[1]) // 2
                color = (0, 255, 0) if self.used[i][j] else (0, 0, 0)
                cv2.putText(image, text, (x, y), font, 0.5, color, 1)
    
        cv2.imshow("Game Status", image)
        cv2.waitKey(1)

    def update_game_state(self):
        """更新游戏状态（每5秒或操作后调用）"""
        frame = self.capture_window()
        processed = self.process_image(frame)
        self.recognize_numbers(processed)
        

    def run(self):
        """改进的主循环"""
        self.update_game_state()  # 初始状态获取
        self.calibrate_grid()  # 标定
        #self.draw_numbers()  # 绘制数字表格
        while True:
            # 处理矩形组合
            for height in range(1, 11):  # 高度从2到3
                for width in range(1, 11):  # 宽度从2到3
                    if self.check_rectangle(height, width):
                        continue  # 如果处理了矩形组合，继续下一轮循环
            # 处理行组合
            if self.check_row():
                continue  # 如果处理了行组合，继续下一轮循环
            time.sleep(0.01)  # 短暂休眠降低 CPU 占用
            # 处理列组合
            if self.check_col():
                continue

if __name__ == "__main__":
    automator = GameAutomator("开局托儿所")
    automator.run()
