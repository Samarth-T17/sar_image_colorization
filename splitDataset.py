import os
import random
import shutil

def create_empty_dirs(root_dir, categories, subdirs):
    for category in categories:
        category_path = os.path.join(root_dir, category)
        os.makedirs(os.path.join(category_path, 's1'), exist_ok=True)
        os.makedirs(os.path.join(category_path, 's2'), exist_ok=True)

def traverse_and_split(src_dir, train_dir, val_dir, val_ratio=0.2):
    def traverse_and_process(current_dir):
        # Check if the current directory has subdirectories
        subdirs = [d for d in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, d))]
        if not subdirs:
            return

        for subdir in subdirs:
            subdir_path = os.path.join(current_dir, subdir)
            traverse_and_process(subdir_path)

        # Process files in the current directory
        files = []
        for subdir in subdirs:
            subdir_path = os.path.join(current_dir, subdir)
            files += [(f, subdir) for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]

        if not files:
            return

        # Split files
        random.seed(42)
        random.shuffle(files)
        split_index = int(len(files) * val_ratio)
        val_files = files[:split_index]
        train_files = files[split_index:]

        # Get current category and subdir names
        category = os.path.basename(current_dir)

        for file, subdir in train_files:
            src_file = os.path.join(current_dir, subdir, file)
            dst_file = os.path.join(train_dir, category, subdir, file)
            shutil.copy(src_file, dst_file)

        for file, subdir in val_files:
            src_file = os.path.join(current_dir, subdir, file)
            dst_file = os.path.join(val_dir, category, subdir, file)
            shutil.copy(src_file, dst_file)

    traverse_and_process(src_dir)

def main():

    # Define categories and subdirectories
    categories = ['agri','barrenland','grassland','urban']  # Add more categories if needed
    subdirs = ['s1', 's2']
    
    source_dir = "v_2"
    train_dir = "train"
    val_dir = "val"

    # Create empty directories in both train and val
    create_empty_dirs(train_dir, categories, subdirs)
    create_empty_dirs(val_dir, categories, subdirs)

    traverse_and_split(source_dir, train_dir, val_dir, val_ratio=0.2)
    
    
    
if __name__ == "__main__":
    main()
