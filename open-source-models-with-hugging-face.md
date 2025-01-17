## 1. 安装 `transformers`
```bash
pip installl transformers
```
## 2. 导入并定义对话管道
```python
from transformers.utils import logging
logging.set_verbosity_error()

# 定义对话管道
from transformers import pipeline
chatbot = pipline(task="conversational",
				 model="./models/facebook/blenderbot-400M-distill")
```
## 3. 使用管道
```python
user_message = """
What are some fun activities I can do in the winter?
"""
```
## 4. 