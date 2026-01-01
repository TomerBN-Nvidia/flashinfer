FROM vllm/vllm-openai:v0.11.2

# Create non-root user with specified UID and GID
# NOTE: Replace with your own UID and GID. This is a workaround from https://github.com/microsoft/vscode-remote-release/issues/49#issuecomment-489060908.
ARG HOST_UID=1003
ARG HOST_GID=1003
RUN groupadd -g $HOST_GID devuser && \
    useradd -m -u $HOST_UID -g $HOST_GID -s /bin/zsh devuser

# Give devuser sudo access
RUN apt-get update && apt-get install -y sudo && \
    echo "devuser ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/devuser && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Install dependencies
RUN apt-get update && apt-get install -y \
    zsh curl git vim tmux && \
    chsh -s $(which zsh) && \
    rm -rf /var/lib/apt/lists/*

# Install Oh My Zsh
RUN git clone https://github.com/ohmyzsh/ohmyzsh.git /root/.oh-my-zsh && \
    cp /root/.oh-my-zsh/templates/zshrc.zsh-template /root/.zshrc

# Add .vimrc
RUN echo "set number"                >> /root/.vimrc && \
    echo "syntax on"                 >> /root/.vimrc && \
    echo 'set tabstop=4'             >> /root/.vimrc && \
    echo 'set shiftwidth=4'          >> /root/.vimrc && \
    echo 'set expandtab'             >> /root/.vimrc && \
    echo 'set autoindent'            >> /root/.vimrc && \
    echo 'set cursorline'            >> /root/.vimrc && \
    echo 'set background=dark'       >> /root/.vimrc

# Add .tmux.conf
RUN echo "set -g mouse on"                       >> /root/.tmux.conf && \
    echo "setw -g mode-keys vi"                  >> /root/.tmux.conf && \
    echo "set -g history-limit 100000"           >> /root/.tmux.conf && \
    echo "set -g prefix C-a"                     >> /root/.tmux.conf && \
    echo "unbind C-b"                            >> /root/.tmux.conf && \
    echo "bind C-a send-prefix"                  >> /root/.tmux.conf && \
    echo "set -g status-bg black"                >> /root/.tmux.conf && \
    echo "set -g status-fg white"                >> /root/.tmux.conf && \
    echo "set -g status-justify centre"          >> /root/.tmux.conf

# Set up oh-my-zsh for devuser
RUN cp -r /root/.oh-my-zsh /home/devuser/.oh-my-zsh && \
    cp /root/.zshrc /home/devuser/.zshrc && \
    cp /root/.vimrc /home/devuser/.vimrc && \
    cp /root/.tmux.conf /home/devuser/.tmux.conf && \
    sed -i 's|/root/.oh-my-zsh|/home/devuser/.oh-my-zsh|g' /home/devuser/.zshrc && \
    chown -R devuser:devuser /home/devuser/

# Set workspace directory and ownership
WORKDIR /sgl-workspace/sglang
RUN chown -R devuser:devuser /sgl-workspace

# Switch to devuser
USER devuser

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y