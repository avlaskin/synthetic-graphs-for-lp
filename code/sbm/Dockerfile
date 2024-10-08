FROM tiagopeixoto/graph-tool:latest

RUN pacman -S wget git sudo --noconfirm --needed
RUN pacman -Syu
RUN pacman -S python-scikit-learn python-networkx --noconfirm --needed
RUN pacman -S python-pip --noconfirm --needed

# For local we need to do that.
# For cloud run - not, as it will use requirements.txt
#RUN useradd builduser -m
#RUN passwd -d builduser
#RUN printf 'builduser ALL=(ALL) ALL\n' | tee -a /etc/sudoers
#RUN wget https://bootstrap.pypa.io/get-pip.py
#RUN python get-pip.py

# USER builduser
#RUN pip install google-cloud
#RUN pip install google-cloud-storage
#RUN pip install gensim

COPY . .
RUN echo "/bin:${PATH}" >> /etc/bash.bashrc
RUN pip install google-cloud --break-system-packages
RUN pip install google-cloud-storage --break-system-packages
RUN pip install google-cloud-secret-manager google-cloud-firestore --break-system-packages

CMD ["python", "main.py"]



