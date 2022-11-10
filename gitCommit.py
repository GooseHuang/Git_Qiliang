from git import Repo
import time

"""
git clone https://github.com/GooseHuang/Git_Test.git

pip install GitPython

# Proxy Setting
git config --global http.proxy http://127.0.0.1:1080
git config --global https.proxy https://127.0.0.1:1080

git config --global --unset http.proxy
git config --global --unset https.proxy


# Set
git config --global http.proxy http://127.0.0.1:1157
git config --global https.proxy https://127.0.0.1:1157


# Unset
git config --global --unset http.proxy
git config --global --unset https.proxy

"""

try:

    PATH_OF_GIT_REPO = r'./.git'  # make sure .git folder is properly configured
    COMMIT_MESSAGE = input("Please enther the commit message:\n")
    # COMMIT_MESSAGE = "j"
    def git_push():
        try:
            repo = Repo(PATH_OF_GIT_REPO)
    #         repo.git.add(update=True)
            repo.git.add(all=True)
            repo.index.commit(COMMIT_MESSAGE)
            origin = repo.remote(name='origin')
            origin.push()
            repo.__del__()
        except Exception as e:
            print('Some error occured while pushing the code:')
            print(e)
            time.sleep(50)

    git_push()
    print("Successed.")
    time.sleep(10)
    
except Exception as e:
    print(e)
    time.sleep(50)