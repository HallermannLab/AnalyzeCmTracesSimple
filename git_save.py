import subprocess
import os
from datetime import datetime
import shutil


def get_git_info():
    def try_command(cmd):
        try:
            return subprocess.check_output(cmd).decode().strip()
        except Exception:
            return "unknown"

    repo_url = try_command(['git', 'remote', 'get-url', 'origin'])
    branch_name = try_command(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    commit_hash = try_command(['git', 'rev-parse', '--short', 'HEAD'])
    commit_msg = try_command(['git', 'log', '-1', '--pretty=%B'])
    commit_date = try_command(['git', 'log', '-1', '--date=iso', '--pretty=%cd'])
    git_status = try_command(['git', 'status', '--short'])

    is_dirty = "dirty" if git_status != "" else "clean"

    return {
        "Repository": repo_url,
        "Branch": branch_name,
        "Commit": commit_hash,
        "Commit message": commit_msg,
        "Commit date": commit_date,
        "Working directory": is_dirty,
        "Modified files": git_status
    }


def save_git_info(output_folder, script_path=None):
    git_info = get_git_info()
    output_path = os.path.join(output_folder, "git_version_info.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        for key, value in git_info.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

    # Optionally, save a copy of the current script
    if script_path and os.path.isfile(script_path):
        try:
            shutil.copy(script_path, os.path.join(output_folder, "analysis_script_backup.py"))
        except Exception as e:
            print(f"Could not copy script: {e}")
