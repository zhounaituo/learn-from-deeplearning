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

# 转换成对话结构
from transformers import Conversation
conversation = Conversation(user_mesage)

# 进行对话
conversation = chatbot(conversation)
print(conversation)

# 追加对话
conversation.add_message(
	{"role": "user",
	 "content": """
What else do you recommend?
"""
	})
conversation = chatbot(conversation)
print(conversation)
```
## 4. 翻译和总结
```bash
pip install transformers
pip install torch
```
### 4.1. 翻译
```python
form transformers.utils import loggin
logging.set_verbosity_error()

form transformers import pipeline
import torch

translator = pipeline(task="translation",
					  model="./model/facebook/nllb-200-distilled-600M",
					  torch_dtype=torch.bfloat16)

text = """\
My puppy is adorable, \
Your kitten is cute.
Her panda is friendly.
His llama is thoughtful. \
We all have nice pets!"""

text_translated = translator(text,
                             src_lang="eng_Latn",
                             tgt_lang="fra_Latn")
```

- [语言选择](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200)
### 4.2. 释放内存
```python
import gc 
del translator 
gc.collect()
```
### 4.3. 总结
```python
summarizer = pipeline(task="summarization",
                      model="./models/facebook/bart-large-cnn",
                      torch_dtype=torch.bfloat16)

text = """Paris is the capital and most populous city of France, with
          an estimated population of 2,175,601 residents as of 2018,
          in an area of more than 105 square kilometres (41 square
          miles). The City of Paris is the centre and seat of
          government of the region and province of Île-de-France, or
          Paris Region, which has an estimated population of
          12,174,880, or about 18 percent of the population of France
          as of 2017."""

summary = summarizer(text,
                     min_length=10,
                     max_length=100)

summary
```
## 5. 句子嵌入
```bash
pip install sentence-transformers
```
### 5.1. 将句子转换为向量表示
```python
# 导入库和设置日志
from transformers.utils import logging
logging.set_verbosity_error()

# 加载模型
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# 定义句子列表
sentences1 = ['The cat sits outside',
              'A man is playing guitar',
              'The movies are awesome']
# 生成嵌入
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings1

sentences2 = ['The dog plays in the garden',
              'A woman watches TV',
              'The new movie is so great']
embeddings2 = model.encode(sentences2, 
                           convert_to_tensor=True)
print(embeddings2)
```
### 5.2. 计算两个句子之间的相似程度
```python
# 导入库并比较两个向量之间的相似程度
from sentence_transformers import util
cosine_scores = util.cos_sim(embeddings1,embeddings2)

print(cosine_scores)

# 打印结果
for i in range(len(sentences1)):
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i],
                                                 sentences2[i],
                                                 cosine_scores[i][i]))
```
## 6. 零样本音频分类
```bash
pip install transformers datasets soundfile librosa
```
### 6.1. 准备录音数据库
```python
from transformers.utils import logging
logging.set_verbosity_error()

# 加载样本
from datasets import load_dataset, load_from_disk

# This dataset is a collection of different sounds of 5 seconds
# dataset = load_dataset("ashraq/esc50",
#                       split="train[0:10]")
dataset = load_from_disk("./models/ashraq/esc50/train")
audio_sample = dataset[0]
audio_sample

# 播放样本
from IPython.display import Audio as IPythonAudio
IPythonAudio(audio_sample["audio"]["array"],
             rate=audio_sample["audio"]["sampling_rate"])
```
### 6.2. 建立音频分类
```python
# 计算模型采样率
from transformers import pipeline
zero_shot_classifier = pipeline(
    task="zero-shot-audio-classification",
    model="./models/laion/clap-htsat-unfused")

# 1s 的高分辨率音频（192000Hz）对于模型（16000Hz）来说相当于 12s
(1 * 192000) / 16000
12.0

# 获取样本采样率
zero_shot_classifier.feature_extractor.sampling_rate

# 设置样本的频率和模型频率相同
from datasets import Audio
dataset = dataset.cast_column(
	"audio",
	Audio(sampling_rate=48_000))

audio_sample = dataset[0]
audio_sample

# 设置分类标签
candidate_labels = ["Sound of a dog",
                    "Sound of vacuum cleaner"]

# 传入样本以及分类标签
zero_shot_classifier(audio_sample["audio"]["array"],
                     candidate_labels=candidate_labels)
```
## 7. 语音识别
```bash
pip install transformers soundfile librosa gradio 
pip install -U datasets
```
### 7.1. 数据准备
```python
# 设置日志级别
from transformers.utils import logging
logging.set_verbosity_error()

# 流式加载 Librispeech 数据集
from datasets import load_dataset
dataset = load_dataset("librispeech_asr",
                       split="train.clean.100",
                       streaming=True,
                       trust_remote_code=True)

# 迭代数据
example = next(iter(dataset))

# 获取前5个数据，并转换为列表
dataset_head = dataset.take(5)
list(dataset_head)

example

# 设置为音频
from IPython.display import Audio as IPythonAudio

IPythonAudio(example["audio"]["array"],
             rate=example["audio"]["sampling_rate"])
```
### 7.2. 识别
```python 
# 设置管道
from transformers import pipeline
asr = pipeline(task="automatic-speech-recognition",
               model="distil-whisper/distil-small.en")

# 查看模型采样率
asr.feature_extractor.sampling_rate

# 查看样本采样率
example['audio']['sampling_rate']

# 识别
asr(example["audio"]["array"])

example["text"]
```
### 7.3. 使用 `Gradio` 构建共享应用
- [Building Generative AI Applications with Gradio](https://www.deeplearning.ai/short-courses/building-generative-ai-applications-with-gradio/)
```python
import os
import gradio as gr

demo = gr.Blocks()

def transcribe_speech(filepath):
    if filepath is None:
        gr.Warning("No audio found, please retry.")
        return ""
    output = asr(filepath)
    return output["text"]

mic_transcribe = gr.Interface(
    fn=transcribe_speech,
    inputs=gr.Audio(sources="microphone",
                    type="filepath"),
    outputs=gr.Textbox(label="Transcription",
                       lines=3),
    allow_flagging="never")
    
file_transcribe = gr.Interface(
    fn=transcribe_speech,
    inputs=gr.Audio(sources="upload",
                    type="filepath"),
    outputs=gr.Textbox(label="Transcription",
                       lines=3),
    allow_flagging="never",
)

with demo:
    gr.TabbedInterface(
        [mic_transcribe,
         file_transcribe],
        ["Transcribe Microphone",
         "Transcribe Audio File"],
    )

demo.launch(share=True, 
            server_port=int(os.environ['PORT1']))

demo.close()
```
### 7.4. 识别长语音
```python
import soundfile as sf
import io

audio, sampling_rate = sf.read('narration_example.wav')

# 查看采样率
sampling_rate
asr.feature_extractor.sampling_rate

# 将音频从立体声转换为单声道
# 返回音频数据形式
audio.shape 

# 置换数据的轴
import numpy as np 
audio_transposed = np.transpose(audio)

audio_transposed.shape

# 转换音频为单声道
import librosa
audio_mono = librosa.to_mono(audio_transposed)

IPythonAudio(audio_mono,
             rate=sampling_rate)
asr(audio_mono)

# 设置音频
sampling_rate
asr.feature_extractor.sampling_rate

audio_16KHz = librosa.resample(audio_mono,
                               orig_sr=sampling_rate,
                               target_sr=16000)
asr(
    audio_16KHz,
    chunk_length_s=30, # 30 seconds
    batch_size=4,
    return_timestamps=True,
)["chunks"]
```
### 7.5. 构建 `Gradio` 共享应用
```python
import gradio as gr 
demo = gr.Blocks()

def transcribe_long_form(filepath):
	if filepath is None:
		gr.Warning("No audio found, please retry.")
		return ""
	output = asr(
		filepath,
		max_new_tokens=256,
		chunk_length_s=30,
		batch_size=8,
	)
	return output["text"]

mic_transcribe = gr.Interface(
	fn=transcribe_long_form,
	inputs=gr.Audio(sources="microphone",
					type="filepath"),
	outputs=gr.Textbox(label="Transcription",
					lines=3),
	allow_flaggine="never"
)

file_transcribe = gr.Interface(
    fn=transcribe_long_form,
    inputs=gr.Audio(sources="upload",
                    type="filepath"),
    outputs=gr.Textbox(label="Transcription",
                       lines=3),
    allow_flagging="never",
)

with demo:
	gr.TabbleInterface(
		[mic_transcribe,
		 file_transcribe],
		["Transcribe Microphone",
		 "Transcribe Audio File"],
	)
demo.launch(share=Ture,
			server_port=int(os.environ['PORT1']))
demo.close()
```
## 8. 文字转语音
```bash
pip install transformers gradio timm inflect phonemizer
sudo apt-get update 
sudo apt-get install espeak-ng
pip install py-espeak-ng
```

```python
from transformers.utils import logging
logging.set_verbosity_error()

from transformers import pipeline
narrator = pipeline("text-to-speech",
                    model="./models/kakao-enterprise/vits-ljs")
# 文字转语音
text = """
Researchers at the Allen Institute for AI, \
HuggingFace, Microsoft, the University of Washington, \
Carnegie Mellon University, and the Hebrew University of \
Jerusalem developed a tool that measures atmospheric \
carbon emitted by cloud servers while training machine \
learning models. After a model’s size, the biggest variables \
were the server’s location and time of day it was active.
"""
narrated_text = narrator(text)

from IPython.display import Audio as IPythonAudio
IPythonAudio(narrated_text["audio"][0],
             rate=narrated_text["sampling_rate"])
```
## 9. 目标检测
### 9.1. 目标检测
```bash
pip install transformers gradio timm inflect phonemizer
sudo apt-get update 
sudo apt-get install espeak-ng
pip install py-espeak-ng
```

```python
from helper import load_image_from_url, render_results_in_image
from transformers import pipeline

from transformers.utils import logging
logging.set_verbosity_error()

from helper import ignore_warings
ignore_warings()

od_pipe = pipeline("object-detection", 
				   "./models/facebook/detr-resnet-50")
# 输入图片材料
from PIL import Image
raw_image = Image.open('huggingface_friends.jpg')
raw_image.resize((569, 491))

# 对象识别
pipeline_output = od_pipe(raw_image)
processed_image = render_results_in_image(
    raw_image, 
    pipeline_output)
    
processed_image
```
### 9.2. 设置 `Gradio` 应用
```python
import os
import gradio as gr

def get_pipeline_prediction(pil_image):
    pipeline_output = od_pipe(pil_image)
    processed_image = render_results_in_image(pil_image,
                                            pipeline_output)
    return processed_image

demo = gr.Interface(
  fn=get_pipeline_prediction,
  inputs=gr.Image(label="Input image", 
                  type="pil"),
  outputs=gr.Image(label="Output image with predicted instances",
                   type="pil")
)

demo.launch(share=True, server_port=int(os.environ['PORT1']))
demo.close()
```
### 9.3. 制作AI语音助手
```python
pipeline_output
od_pipe

raw_image = Image.open('huggingface_friends.jpg')
raw_image.resize((284,245))

# 图片转自然语言
from helper import summarize_predictions_natural_language
text = summarize_predictions_natural_language(pipeline_output)
text

# 文字转音频
tts_pipe = pipeline("text-to-speech",
                    model="./models/kakao-enterprise/vits-ljs")
tts_pipe = pipeline("text-to-speechhttps://s172-29-42-60p8888.lab-aws-production.deeplearning.ai/notebooks/L08/L8_object_detection.ipynb#Generate-Audio-Narration-of-an-Image",
                    model="./models/kakao-enterprise/vits-ljs")
narrated_text = tts_pipe(text)

from IPython.display import Audio as IPythonAudio
IPythonAudio(narrated_text["audio"][0],
             rate=narrated_text["sampling_rate"])
```
## 10. 图片分割
```bash
pip install transformers gradio timm torchvision
```

```python
from transformers.utils import logging
logging.set_verbosity_error()
```
### 10.1. 使用 SAM 生成分割掩码
```python
from transformers import pipeline
sam_pipe = pipeline("mask-generation",
    "./models/Zigeng/SlimSAM-uniform-77")

# 加载数据
from PIL import Image
raw_image = Image.open('meta_llamas.jpg')
raw_image.resize((720, 375))

# 分割生成掩码
output = sam_pipe(raw_image, points_per_batch=32)
from helper import show_pipe_masks_on_image
show_pipe_masks_on_image(raw_image, output)
```
### 10.2. 更快的推理：推理图像和单点
```python
# 导入和加载模型
from transformers import SamModel, SamProcessor

model = SamModel.from_pretrained(
    "./models/Zigeng/SlimSAM-uniform-77")
processor = SamProcessor.from_pretrained(
    "./models/Zigeng/SlimSAM-uniform-77")

# 调整图片大小
raw_image.resize((720, 375))

# 定义和处理输入点，生成分割掩码
input_points = [[[1600, 700]]]
inputs = processor(raw_image,
                 input_points=input_points,
                 return_tensors="pt")

import torch

# 禁止梯度计算
with torch.no_grad():
	outputs = model(**inputs)

# 处理分割掩码
predicted_masks = processor.image_processor.post_process_masks(
    outputs.pred_masks,
    inputs["original_sizes"],
    inputs["reshaped_input_sizes"]
)

# 对应图片使用数量
len(predicted_masks)

# 获取预测掩码的大小
predicted_mask = predicted_masks[0]
predicted_mask.shape

# 显示分割掩码
outputs.iou_scores

from helper import show_mask_on_image
for i in range(3):
    show_mask_on_image(raw_image, predicted_mask[:, i])
```

### 10.3. 基于 DPT 的深度评估
```python
depth_estimator = pipeline(task="depth-estimation",
                        model="./models/Intel/dpt-hybrid-midas")

raw_image = Image.open('gradio_tamagochi_vienna.png')
raw_image.resize((806, 621))

output = depth_estimator(raw_image)
output

# 调整大小
output["predicted_depth"].shape
output["predicted_depth"].unsqueeze(1).shape
prediction = torch.nn.functional.interpolate(
	output["predicted_depth"].unsqueeze(1),
	size=raw_image.size[::-1],
	mode="bicubic",
	align_corners=False,
)
prediction.shape
raw_image.size[::-1],
prediction

# 标准化张量以显示
import numpy as np
output = prediction.squeeze().numpy()
formatted = (output * 255 / np.max(output)).astype("uint8")
depth = Image.fromarray(formated)
depth
```

### 10.4. 使用 `Gradio`
```python
import os
import gradio as gr
from transformers import pipeline

def launch(input_image):
    out = depth_estimator(input_image)

    # resize the prediction
    prediction = torch.nn.functional.interpolate(
        out["predicted_depth"].unsqueeze(1),
        size=input_image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # normalize the prediction
    output = prediction.squeeze().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)
    return depth

iface = gr.Interface(launch, 
                     inputs=gr.Image(type='pil'), 
                     outputs=gr.Image(type='pil'))

iface.launch(share=True, server_port=int(os.environ['PORT1']))

iface.close()
```
## 11. 图片搜索
```bash
pip install transformers torch
```

```python
# 设置提示等级
from transformers.utils import logging
logging.set_verbosity_error()

# 配置模型
from transformers import BlipForImageTextRetrieval
model = BlipForImageTextRetrieval.from_pretrained(
    "./models/Salesforce/blip-itm-base-coco")

from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(
    "./models/Salesforce/blip-itm-base-coco")

# 配置资源
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'

from PIL import Image
import requests

raw_image =  Image.open(
    requests.get(img_url, stream=True).raw).convert('RGB')
raw_image
```
### 11.1. 测试文本和图片是否匹配
```python
text = "an image of a woman and a dog on the beach"
inputs = processor(images=raw_image,
                   text=text,
                   return_tensors="pt")
inputs

itm_scores = model(**inputs)[0]
itm_scores

import torch

# 使用 softmax layer 获取概率
itm_score = torch.nn.functional.softmax(
    itm_scores,dim=1)
itm_score
print(f"""\
The image and text are matched \
with a probability of {itm_score[0][1]:.4f}""")
```
## 12. 图像说明
```bash
pip install transformers
```

```python
# 设置警告等级
from transformers.utils import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings("ignore", message="Using the model-agnostic default `max_length`")

# 设置模型
from transformers import BlipForConditionalGeneration
model = BlipForConditionalGeneration.from_pretrained(
    "./models/Salesforce/blip-image-captioning-base")

from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(
    "./models/Salesforce/blip-image-captioning-base")

# 加载图片资源
from PIL import Image
image = Image.open("./beach.jpeg")
image
```
### 12.1. 条件图片说明
```python
text = "a photograph of"
inputs = processor(image, text, return_tensors="pt")
inputs

out = model.generate(**inputs)
out

print(processor.decode(out[0], skip_special_tokens=True))
```
### 12.2. 无条件图片说明
```python
inputs = processor(image,return_tensors="pt")

out = model.generate(**inputs)
out 

print(processor.decode(out[0], skip_special_tokens=True))
```
## 13. 视觉回答
```bash
pip install transformers
```

```python 
# 配置提示信息
from transformers.utils import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings("ignore", message="Using the model-agnostic default `max_length`")

# 加载模型和进程
from transformers import BlipForQuestionAnswering
model = BlipForQuestionAnswering.from_pretrained(
    "./models/Salesforce/blip-vqa-base")

from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(
    "./models/Salesforce/blip-vqa-base")

# 加载图片资源
from PIL import Image
image = Image.open("./beach.jpeg")
image

# 问答
question = "how many dogs are in the picture?"
inputs = processor(image, question, return_tensors="pt")
out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
```
## 14. 零样本图片分类
```bash 
pip install transformers
```

```python
from transformers.utils import logging
logging.set_verbosity_error()

# 加载模型和过程
from transformers import CLIPModel
model = CLIPModel.from_pretrained(
    "./models/openai/clip-vit-large-patch14")

from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(
    "./models/openai/clip-vit-large-patch14")

# 加载图片
from PIL import Image
image = Image.open("./kittens.jpeg")
image

# 分类标签
labels = ["a photo of a cat", "a photo of a dog"]
inputs = processor(text=labels,
                   images=image,
                   return_tensors="pt",
                   padding=True)

outputs = model(**inputs)
outputs

outputs.logits_per_image

probs = outputs.logits_per_image.softmax(dim=1)[0]
probs

probs = list(probs)
for i in range(len(labels)):
  print(f"label: {labels[i]} - probability of {probs[i].item():.4f}")
```
## 15. 在 `Gradio` 上部署 ML 模型
```bash
pip install transformers gradio gradio_client
pip install -U gradio_client
```

```python
# 配置警告等级
from transformers.utils import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings("ignore", 
                        message="Using the model-agnostic default `max_length`")
```
### 15.1. 图片字幕
```python
import os
import gradio as gr

from transformers import pipeline
pipe = pipeline("image-to-text",
                model="./models/Salesforce/blip-image-captioning-base")

def launch(input):
    out = pipe(input)
    return out[0]['generated_text']

iface = gr.Interface(launch,
                     inputs=gr.Image(type='pil'),
                     outputs="text")

iface.launch(share=True, 
             server_port=int(os.environ['PORT1']))
```