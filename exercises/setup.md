# Clean ubuntu installation

1. Install latest Ubuntu LTS version

2. Install git:

    ```sh
    sudo apt install git
    ```

3. Setup:

    ```sh
    git clone https://github.com/endikallanomatxin/setup
    cd setup
    bash setup.sh
    ```

4. Clone our repo:

    ```sh
    cd ~/Documents
    git clone https://github.com/endikallanomatxin/hackathon
    ```

5. Login into github:

    ```sh
    sudo apt install gh
    gh auth login
    ```

6. Setup git name and email

    ```sh
    git config --global user.email "your@email.com"
    git config --global user.name "Yourname Yourlastname"
    ```

7. Install python 3.10:

    ```sh
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install python3.10 python3.10-venv python3.10-dev
    ```

8. Create a python 3.10 environment:

    ```sh
    python3.10 -m venv .venv
    source .venv/bin/activate
    cd exercises/genesis
    pip install -r requirements.txt
    ```

9. Try to run an example:

    ```sh
    python 01_rl/train.py
    ```

