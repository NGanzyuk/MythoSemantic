import subprocess
import sys


def run_script(script_name):
    print(f"🚀 Running {script_name}...")
    result = subprocess.run([sys.executable, script_name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ {script_name} failed!")
        print(result.stderr)
        sys.exit(1)
    else:
        print(f"✅ {script_name} completed successfully.\n")


if __name__ == "__main__":
    # Этап 1: Сборка корпуса
    run_script("download_and_build_corpus.py")

    # Этап 2: Построение эмбеддингов
    run_script("embedding_builder.py")

    print("🎉 Project pipeline completed!")
