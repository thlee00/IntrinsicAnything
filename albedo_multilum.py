import os
import shutil
import tempfile
import subprocess

input_path = "/workspace/controlnet-diffusers-relighting/multilum_images/1536x1024"
output_path = "out/albedo"
os.makedirs(output_path, exist_ok=True)

# 파일 필터링 함수
def filter_files(input_path, suffix):
    with os.scandir(input_path) as entries:
        return [entry.path for entry in entries if entry.is_file() and entry.name.endswith(suffix)]

# 파일을 임시 디렉토리에 복사
def copy_files_to_temp_dir(file_paths):
    temp_dir = tempfile.mkdtemp()
    for file_path in file_paths:
        shutil.copy(file_path, temp_dir)
    return temp_dir

# inference.py 실행 함수
def process_images(filtered_input_dir, num_files):
    print(f"Processing {num_files} files using inference.py...")
    command = [
        "python", "inference.py",
        "--input_dir", filtered_input_dir,
        "--model_dir", "weights/albedo",
        "--output_dir", output_path,
        "--ddim", "100",
        "--batch_size", "4"
    ]
    subprocess.run(command)
    print("Completed processing selected images.")

# 메인 함수
def main():
    # _dir_10.png로 끝나는 파일 필터링
    filtered_files = filter_files(input_path, "_dir_10.png")
    num_files = len(filtered_files)
    print(f"Found {num_files} files matching '_dir_10.png'.")

    if num_files == 0:
        print("No files to process.")
        return

    # 파일을 임시 디렉토리에 복사
    temp_dir = copy_files_to_temp_dir(filtered_files)
    print(f"Copied {num_files} files to temporary directory: {temp_dir}")

    # inference.py 실행
    try:
        process_images(temp_dir, num_files)
    finally:
        shutil.rmtree(temp_dir)
        print(f"Temporary directory {temp_dir} has been removed.")

if __name__ == "__main__":
    main()
