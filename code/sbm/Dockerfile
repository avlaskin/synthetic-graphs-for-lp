FROM tiagopeixoto/graph-tool:latest
RUN pacman -S wget git sudo --noconfirm --needed
RUN pacman -Syu
RUN pacman -S python-scikit-learn python-networkx python-pipx --noconfirm --needed

# USER builduser
RUN pipx install google-cloud
RUN pipx install google-cloud-storage
RUN pipx install gensim

COPY . .

CMD ["python" "main.py"]



