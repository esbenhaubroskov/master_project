import json
import os
import subprocess

def get_branch():
    branch = subprocess.check_output("git rev-parse --abbrev-ref HEAD", shell=True)
    return branch.decode()

def get_commit_hash():
    commit_hash = subprocess.check_output("git rev-parse HEAD", shell=True)
    return commit_hash.decode()

def log_git_info(open_path, save_path):
    branch = get_branch()
    commit_hash = get_commit_hash()
    
    with open(open_path) as f:
        config = json.load(f)
        config['git_info']['branch'] = branch.strip()
        config['git_info']['commit_hash'] = commit_hash.strip()
        json.dump(config, open(save_path, "w"), indent = 4) 



if __name__ == '__main__':
    print(get_branch())
    print(get_commit_hash())
    open_path = r'C:\Users\esben\Documents\master-project\RMP\rmp-project\utils\config_test.json'
    save_path = r'C:\Users\esben\Documents\master-project\RMP\rmp-project\utils\config_test_1.json'
    log_git_info(open_path, save_path)
