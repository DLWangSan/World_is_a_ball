import re

# 定义一个函数来处理文件内容
def process_file(input_file, output_file):
    # 使用正则表达式匹配并提取tensor中的数字
    pattern = re.compile(r'tensor\(([-+]?\d*\.\d+|\d+), device=\'cuda:0\', grad_fn=<DivBackward0>\)')

    with open(input_file, 'r') as f:
        lines = f.readlines()

    # 替换所有匹配的字符串
    new_lines = [pattern.sub(r'\1', line) for line in lines]

    # 将处理后的内容写入新文件
    with open(output_file, 'w') as f:
        f.writelines(new_lines)

# 假设原始文件名和目标文件名
input_file = '/runs/2/train_epoch.txt'  # 替换为实际的输入文件路径
output_file = '/models/2/train_epoch.txt'  # 替换为期望的输出文件路径

# 调用函数处理文件
process_file(input_file, output_file)

print("File processing completed. The output is saved to:", output_file)
