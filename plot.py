import re

import matplotlib.pyplot as plt
# 定义读取和处理日志文件的函数
def extract_round_values(log_file):
    # 正则表达式模式，用于提取数值
    pattern = r"round\s\d+:\s([0-9]*\.[0-9]+)"
    
    # 存储提取的数值
    extracted_values = []

    # 逐行读取日志文件
    with open(log_file, "r") as file:
        for line in file:
            # 使用正则表达式搜索符合条件的行
            match = re.search(pattern, line)
            if match:
                # 提取并存储匹配的数值
                extracted_values.append(float(match.group(1)))
    
    return extracted_values

# 定义日志文件的路径
log_file_path = '/Users/jiangdesheng/Desktop/hkust_flower/test/server.log'  # 修改为实际文件路径

# 调用函数处理日志文件
extracted_values = extract_round_values(log_file_path)

# 打印提取的数值
new_extracted_values = [extracted_values[i] for i in range(len(extracted_values)) if i % 2 == 0]
print(new_extracted_values)
print(len(new_extracted_values))

# 创建折线图
plt.figure(figsize=(10, 6))
plt.plot(new_extracted_values, marker='o', linestyle='-', color='g')

# 添加图表标题和坐标轴标签
plt.title('FedProx on multi machine')
plt.xlabel('round')
plt.ylabel('loss')

# 显示网格
plt.grid(True)

# 显示图表
plt.show()