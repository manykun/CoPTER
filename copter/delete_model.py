import os

def delete_files_with_prefix(folder_path, prefix):
    """
    删除指定文件夹中所有以特定前缀开头的文件
    
    参数:
        folder_path: 文件夹路径
        prefix: 要匹配的文件前缀
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return
    
    # 获取文件夹中所有文件
    files = os.listdir(folder_path)
    
    # 筛选出符合前缀的文件
    matching_files = [f for f in files if f.startswith(prefix) and os.path.isfile(os.path.join(folder_path, f))]
    
    if not matching_files:
        print(f"没有找到以 '{prefix}' 为前缀的文件")
        return
    
    # 显示找到的文件并确认删除
    print(f"找到以下 {len(matching_files)} 个文件:")
    for file in matching_files:
        print(f"  - {file}")
    
    confirm = input("确定要删除这些文件吗? (y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消删除操作")
        return
    
    # 执行删除操作
    deleted_count = 0
    for file in matching_files:
        file_path = os.path.join(folder_path, file)
        try:
            os.remove(file_path)
            deleted_count += 1
            print(f"已删除: {file}")
        except Exception as e:
            print(f"删除 {file} 失败: {str(e)}")
    
    print(f"操作完成，共删除 {deleted_count} 个文件")

if __name__ == "__main__":
    # 示例用法
    folder = "/home/ame/copter/copter/models/"  # 替换为你的文件夹路径
    prefix = "experiment_acc_ACC_"  # 替换为你要匹配的前缀
    
    delete_files_with_prefix(folder, prefix)