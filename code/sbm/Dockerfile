FROM tiagopeixoto/graph-tool:latest

RUN pacman -S wget git sudo --noconfirm --needed
RUN pacman -Syu
RUN pacman -S python-scikit-learn python-networkx python-pip --noconfirm --needed

COPY . .
RUN echo "/bin:${PATH}" >> /etc/bash.bashrc
RUN pip install google-cloud google-cloud-storage google-cloud-secret-manager google-cloud-firestore --break-system-packages

CMD ["python", "main.py"]



