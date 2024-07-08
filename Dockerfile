FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
EXPOSE 15556
CMD ["python", "app.py"]
