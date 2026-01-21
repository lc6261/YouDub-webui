import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import subprocess
import sys
import os
import time
import csv

class YouDubGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YouDub - YouTube视频自动翻译配音工具")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # 设置主题
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # 创建主框架
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建笔记本（标签页）
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 创建处理选项标签页
        self.process_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.process_tab, text="处理选项")
        
        # 创建CSV管理标签页
        self.csv_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.csv_tab, text="CSV管理")
        
        # 处理选项标签页内容
        # 创建输入区域
        self.input_frame = ttk.LabelFrame(self.process_tab, text="输入设置", padding="10")
        self.input_frame.pack(fill=tk.X, pady=5)
        
        # 视频URL输入
        ttk.Label(self.input_frame, text="视频URL或CSV文件路径:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.url_var = tk.StringVar()
        self.url_entry = ttk.Entry(self.input_frame, textvariable=self.url_var, width=60)
        self.url_entry.grid(row=0, column=1, sticky=tk.W, pady=5)
        self.browse_button = ttk.Button(self.input_frame, text="浏览", command=self.browse_file)
        self.browse_button.grid(row=0, column=2, sticky=tk.W, pady=5, padx=5)
        
        # 处理选项区域
        self.options_frame = ttk.LabelFrame(self.process_tab, text="处理选项", padding="10")
        self.options_frame.pack(fill=tk.X, pady=5)
        
        # 起始步骤
        ttk.Label(self.options_frame, text="起始步骤:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.step_var = tk.StringVar(value="1")
        self.step_combo = ttk.Combobox(self.options_frame, textvariable=self.step_var, values=["1", "2", "3", "4", "5", "6", "7"], width=5)
        self.step_combo.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # 结束步骤
        ttk.Label(self.options_frame, text="结束步骤:").grid(row=0, column=2, sticky=tk.W, pady=5)
        self.end_var = tk.StringVar(value="7")
        self.end_combo = ttk.Combobox(self.options_frame, textvariable=self.end_var, values=["1", "2", "3", "4", "5", "6", "7"], width=5)
        self.end_combo.grid(row=0, column=3, sticky=tk.W, pady=5)
        
        # 模型选择
        ttk.Label(self.options_frame, text="模型:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.model_var = tk.StringVar(value="medium")
        self.model_combo = ttk.Combobox(self.options_frame, textvariable=self.model_var, values=["tiny", "small", "medium", "large", "large-v2"], width=10)
        self.model_combo.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # 语言选择
        ttk.Label(self.options_frame, text="目标语言:").grid(row=1, column=2, sticky=tk.W, pady=5)
        self.lang_var = tk.StringVar(value="中文")
        self.lang_combo = ttk.Combobox(self.options_frame, textvariable=self.lang_var, values=["中文", "英语", "日语", "韩语"], width=10)
        self.lang_combo.grid(row=1, column=3, sticky=tk.W, pady=5)
        
        # 速度调整
        ttk.Label(self.options_frame, text="播放速度:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.speed_var = tk.StringVar(value="1.0")
        self.speed_entry = ttk.Entry(self.options_frame, textvariable=self.speed_var, width=10)
        self.speed_entry.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # 无字幕选项
        self.no_subtitles_var = tk.BooleanVar(value=False)
        self.no_subtitles_check = ttk.Checkbutton(self.options_frame, text="无字幕", variable=self.no_subtitles_var)
        self.no_subtitles_check.grid(row=2, column=2, sticky=tk.W, pady=5)
        
        # 文件夹选择
        ttk.Label(self.options_frame, text="输出文件夹:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.folder_var = tk.StringVar(value="videos")
        self.folder_entry = ttk.Entry(self.options_frame, textvariable=self.folder_var, width=40)
        self.folder_entry.grid(row=3, column=1, sticky=tk.W, pady=5)
        self.folder_button = ttk.Button(self.options_frame, text="浏览", command=self.browse_folder)
        self.folder_button.grid(row=3, column=2, sticky=tk.W, pady=5, padx=5)
        
        # 控制按钮区域
        self.controls_frame = ttk.Frame(self.process_tab)
        self.controls_frame.pack(fill=tk.X, pady=5)
        
        self.start_button = ttk.Button(self.controls_frame, text="开始处理", command=self.start_process)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.pause_button = ttk.Button(self.controls_frame, text="暂停", command=self.pause_process, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(self.controls_frame, text="停止", command=self.stop_process, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # 进度条
        self.progress_frame = ttk.Frame(self.process_tab)
        self.progress_frame.pack(fill=tk.X, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5)
        
        self.progress_label = ttk.Label(self.progress_frame, text="准备就绪")
        self.progress_label.pack(side=tk.RIGHT, padx=5)
        
        # 日志输出区域
        self.log_frame = ttk.LabelFrame(self.process_tab, text="日志输出", padding="10")
        self.log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = tk.Text(self.log_frame, wrap=tk.WORD, height=15)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        self.scrollbar = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=self.scrollbar.set)
        
        # CSV管理标签页内容
        self.csv_control_frame = ttk.Frame(self.csv_tab)
        self.csv_control_frame.pack(fill=tk.X, pady=5)
        
        self.load_csv_button = ttk.Button(self.csv_control_frame, text="加载CSV文件", command=self.load_csv)
        self.load_csv_button.pack(side=tk.LEFT, padx=5)
        
        self.save_csv_button = ttk.Button(self.csv_control_frame, text="保存CSV文件", command=self.save_csv, state=tk.DISABLED)
        self.save_csv_button.pack(side=tk.LEFT, padx=5)
        
        self.add_row_button = ttk.Button(self.csv_control_frame, text="添加行", command=self.add_row, state=tk.DISABLED)
        self.add_row_button.pack(side=tk.LEFT, padx=5)
        
        self.delete_row_button = ttk.Button(self.csv_control_frame, text="删除行", command=self.delete_row, state=tk.DISABLED)
        self.delete_row_button.pack(side=tk.LEFT, padx=5)
        
        # CSV表格区域
        self.csv_frame = ttk.LabelFrame(self.csv_tab, text="CSV内容", padding="10")
        self.csv_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 创建表格
        self.csv_tree = ttk.Treeview(self.csv_frame)
        self.csv_tree.pack(fill=tk.BOTH, expand=True)
        
        # 添加滚动条
        self.csv_scrollbar_y = ttk.Scrollbar(self.csv_tree, orient=tk.VERTICAL, command=self.csv_tree.yview)
        self.csv_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.csv_tree.configure(yscrollcommand=self.csv_scrollbar_y.set)
        
        self.csv_scrollbar_x = ttk.Scrollbar(self.csv_tab, orient=tk.HORIZONTAL, command=self.csv_tree.xview)
        self.csv_scrollbar_x.pack(fill=tk.X, pady=5)
        self.csv_tree.configure(xscrollcommand=self.csv_scrollbar_x.set)
        
        # CSV文件路径
        self.current_csv_path = None
        self.csv_data = []
        
        # 状态变量
        self.process = None
        self.is_paused = False
        self.is_running = False
        
        # 重定向标准输出到日志窗口
        sys.stdout = self
        sys.stderr = self
    
    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV文件", "*.csv"), ("所有文件", "*")])
        if file_path:
            self.url_var.set(file_path)
    
    def browse_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.folder_var.set(folder_path)
    
    def start_process(self):
        # 检查输入
        url = self.url_var.get().strip()
        if not url:
            messagebox.showerror("错误", "请输入视频URL或CSV文件路径")
            return
        
        # 构建命令，使用虚拟环境中的Python解释器
        python_exe = os.path.join("venv", "Scripts", "python.exe")
        cmd = [python_exe, "-m", "youdub.do_everything"]
        
        # 添加参数
        cmd.extend(["--step", self.step_var.get()])
        cmd.extend(["--end", self.end_var.get()])
        cmd.extend(["--folder", self.folder_var.get()])
        cmd.extend(["--model", self.model_var.get()])
        cmd.extend(["--lang", self.lang_var.get()])
        cmd.extend(["--speed", self.speed_var.get()])
        
        if self.no_subtitles_var.get():
            cmd.append("--no-subtitles")
        
        if url.endswith(".csv"):
            cmd.extend(["--url", url])
        else:
            cmd.extend(["--url", url])
        
        # 打印命令信息
        print(f"执行命令: {' '.join(cmd)}")
        print(f"当前工作目录: {os.path.dirname(os.path.abspath(__file__))}")
        print(f"Python解释器: {python_exe}")
        
        # 更新界面状态
        self.start_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL)
        self.progress_label.config(text="开始处理...")
        
        # 启动处理线程
        self.is_running = True
        self.is_paused = False
        
        def run_process():
            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=os.path.dirname(os.path.abspath(__file__))
                )
                
                for line in iter(self.process.stdout.readline, ''):
                    if not self.is_running:
                        break
                    print(line.strip())
                
                self.process.wait()
                
                if self.process.returncode == 0:
                    self.progress_label.config(text="处理完成")
                    messagebox.showinfo("成功", "视频处理完成！")
                else:
                    self.progress_label.config(text="处理失败")
                    messagebox.showerror("错误", f"处理失败，返回码: {self.process.returncode}")
            except Exception as e:
                self.progress_label.config(text="处理失败")
                messagebox.showerror("错误", f"处理失败: {str(e)}")
            finally:
                self.is_running = False
                self.start_button.config(state=tk.NORMAL)
                self.pause_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.DISABLED)
                self.progress_var.set(0)
        
        self.thread = threading.Thread(target=run_process)
        self.thread.daemon = True
        self.thread.start()
    
    def pause_process(self):
        if self.process:
            self.is_paused = not self.is_paused
            if self.is_paused:
                self.process.send_signal(subprocess.signal.SIGSTOP)
                self.pause_button.config(text="继续")
                self.progress_label.config(text="已暂停")
            else:
                self.process.send_signal(subprocess.signal.SIGCONT)
                self.pause_button.config(text="暂停")
                self.progress_label.config(text="继续处理...")
    
    def stop_process(self):
        if self.process:
            self.is_running = False
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.progress_label.config(text="已停止")
    
    def write(self, text):
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)
    
    def flush(self):
        pass
    
    def load_csv(self):
        """加载CSV文件并显示其内容"""
        file_path = filedialog.askopenfilename(filetypes=[("CSV文件", "*.csv"), ("所有文件", "*")])
        if not file_path:
            return
        
        try:
            # 清空现有数据
            for item in self.csv_tree.get_children():
                self.csv_tree.delete(item)
            
            self.csv_data = []
            
            # 读取CSV文件
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader)
                
                # 设置表格列
                self.csv_tree['columns'] = headers
                self.csv_tree['show'] = 'headings'
                
                # 添加列标题
                for header in headers:
                    self.csv_tree.heading(header, text=header)
                    self.csv_tree.column(header, width=100, anchor=tk.W)
                
                # 添加数据行
                for row in reader:
                    self.csv_data.append(row)
                    self.csv_tree.insert('', tk.END, values=row)
            
            self.current_csv_path = file_path
            self.save_csv_button.config(state=tk.NORMAL)
            self.add_row_button.config(state=tk.NORMAL)
            self.delete_row_button.config(state=tk.NORMAL)
            
            messagebox.showinfo("成功", f"成功加载CSV文件: {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("错误", f"加载CSV文件失败: {str(e)}")
    
    def save_csv(self):
        """保存CSV文件"""
        if not self.current_csv_path:
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV文件", "*.csv"), ("所有文件", "*")])
            if not file_path:
                return
            self.current_csv_path = file_path
        
        try:
            with open(self.current_csv_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                
                # 写入表头
                headers = self.csv_tree['columns']
                writer.writerow(headers)
                
                # 写入数据
                for item in self.csv_tree.get_children():
                    row = self.csv_tree.item(item, 'values')
                    writer.writerow(row)
            
            messagebox.showinfo("成功", f"成功保存CSV文件: {os.path.basename(self.current_csv_path)}")
            
        except Exception as e:
            messagebox.showerror("错误", f"保存CSV文件失败: {str(e)}")
    
    def add_row(self):
        """添加新行"""
        if not self.csv_tree['columns']:
            messagebox.showerror("错误", "请先加载CSV文件")
            return
        
        # 创建一个新的空行
        headers = self.csv_tree['columns']
        new_row = [''] * len(headers)
        
        # 添加到表格
        self.csv_tree.insert('', tk.END, values=new_row)
        
    def delete_row(self):
        """删除选中的行"""
        selected_items = self.csv_tree.selection()
        if not selected_items:
            messagebox.showerror("错误", "请先选择要删除的行")
            return
        
        for item in selected_items:
            self.csv_tree.delete(item)
        
        messagebox.showinfo("成功", f"成功删除 {len(selected_items)} 行")

if __name__ == "__main__":
    root = tk.Tk()
    app = YouDubGUI(root)
    root.mainloop()