import os
import shutil
import tempfile
import subprocess

# 경로 설정
input_path = "/workspace/controlnet-diffusers-relighting/multilum_images/1536x1024"
output_path = "out/albedo_high_res"
os.makedirs(output_path, exist_ok=True)

# 파일 필터링 함수
def filter_files(input_path, suffix):
    """
    특정 조건에 맞는 파일을 필터링합니다.
    """
    with os.scandir(input_path) as entries:
        return [entry.path for entry in entries if entry.is_file() and entry.name.endswith(suffix)]

# 파일을 임시 디렉토리에 복사
def copy_files_to_temp_dir(file_paths):
    """
    파일들을 임시 디렉토리에 복사합니다.
    """
    temp_dir = tempfile.mkdtemp()
    for file_path in file_paths:
        shutil.copy(file_path, temp_dir)
    return temp_dir

# inference.py 실행 함수
def process_images_with_custom_command(filtered_input_dir, num_files):
    """
    주어진 커맨드로 inference.py를 실행합니다.
    """
    print(f"Processing {num_files} files using inference.py with custom command...")
    command = [
        "python", "inference.py",
        "--input_dir", filtered_input_dir,
        "--model_dir", "weights/albedo",
        "--output_dir", output_path,
        "--ddim", "100",
        "--batch_size", "8",
        "--guidance_dir", "out/albedo",
        "--guidance", "3",
        "--splits_vertical", "2",
        "--splits_horizontal", "2",
        "--splits_overlap", "1"
    ]
    subprocess.run(command)
    print("Completed processing selected images with custom command.")

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
        process_images_with_custom_command(temp_dir, num_files)
    finally:
        # 임시 디렉토리 삭제
        shutil.rmtree(temp_dir)
        print(f"Temporary directory {temp_dir} has been removed.")

if __name__ == "__main__":
    main()
