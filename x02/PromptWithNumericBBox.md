# u02 Prompt 示例（有题面数值框）

示例问题与根目录 [v0-1PromptWithNumericBBox.md](../v0-1PromptWithNumericBBox.md) 相同。以下示例按当前 `src/eval/infer_xlrs_lite.py` 和对应运行脚本构造。

> 路由：有题面框时，stage1 仍将题面 bbox 映射到全局图坐标，但 crop 统一使用 stage1 输出的 `model_bbox_raw`；stage2 使用当前 u02 的 global image + crop image 构造。该批次对应根目录批处理脚本：`../run_u_reference_50x3.sh`。

## Stage 1

### 构造代码

```python
stage1_prompt = make_stage1_prompt(
    question_prompt, len(global_images), doc
)
stage1_images = global_images
```

图像输入：1 张下采样全局图。

### 实际 Prompt

~~~text
<|im_start|>system
You are a helpful assistant. <|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Identify the object category within the given bounding box in the satellite or aerial image. Image resolution: 1024 x 1024. Bounding box: [25.1904, 656.9984, 59.1872, 704.2048].

The choices are listed below:
(A) The Purple Car
(B) The Blue Forest
(C) Green Forests
(D) Fields of green crops
You are an intelligent remote sensing analyst.
Given a natural language question about a satellite image, generate a structured reasoning answer as follows:
1. <think> ... </think>
    - Provide a neutral one-sentence description of the whole image scene.
    - Cropping task: "This question is asking about <short intent>, therefore I need to crop the image to examine the surroundings of the mentioned target."
    - Non-cropping task: "This question is asking about <short intent>, therefore I need to analyze the entire image without cropping."
    - Include:
        * Question Intent: describe the type of question (object category, spatial relation, count, etc.) and needed visual info.
        * Localization Strategy:
            - Cropping: approximate referent object location in natural language (no coordinates).
            - Non-cropping: strategy to detect all relevant objects.      * Reasoning Result:
    - Cropping: output exactly one JSON-formatted bbox for the referent:          [{"bbox_2d": [x_min,y_min,x_max,y_max], "label": "<short description>"}]
    - Non-cropping: summarize how detected objects will be used to produce the count.
2. <think> ... </think> (only when saw the cropped image)
    - Explain how to reason step by step from the referent (or detected objects) to the final answer. 
3. <answer> ... </answer>
    - Your final answer, strictly following the format requirements in the XLRS answer format below.
Rules: 
    - Always return exactly one <answer> block, for tasks that need cropping, you can provide the bounding box of the object you are intrested, after given the cropped image, you can generate another <think> block to find the answer. 
    - For cropping tasks, also include a bounidng box in <stage_2_reasoning> block 
    - If unsure about localization, make a best guess—never say uncertain.

XLRS answer format:
Select exactly one best option. Inside <answer>, output exactly one uppercase letter from ABCD, with no other text.
<|im_end|><|im_start|>assistant
~~~

## Stage 2

### 构造代码

```python
stage2_prompt = make_stage2_prompt(
    stage1_prompt, stage1_output, len(crop_images)
)
stage2_images = global_images + crop_images
```

图像输入：1 张下采样全局图 + 1 张高分辨率裁剪图。

### 实际 Prompt

~~~text
<|im_start|>system
You are a helpful assistant. <|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Identify the object category within the given bounding box in the satellite or aerial image. Image resolution: 1024 x 1024. Bounding box: [25.1904, 656.9984, 59.1872, 704.2048].

The choices are listed below:
(A) The Purple Car
(B) The Blue Forest
(C) Green Forests
(D) Fields of green crops
You are an intelligent remote sensing analyst.
Given a natural language question about a satellite image, generate a structured reasoning answer as follows:
1. <think> ... </think>
    - Provide a neutral one-sentence description of the whole image scene.
    - Cropping task: "This question is asking about <short intent>, therefore I need to crop the image to examine the surroundings of the mentioned target."
    - Non-cropping task: "This question is asking about <short intent>, therefore I need to analyze the entire image without cropping."
    - Include:
        * Question Intent: describe the type of question (object category, spatial relation, count, etc.) and needed visual info.
        * Localization Strategy:
            - Cropping: approximate referent object location in natural language (no coordinates).
            - Non-cropping: strategy to detect all relevant objects.      * Reasoning Result:
    - Cropping: output exactly one JSON-formatted bbox for the referent:          [{"bbox_2d": [x_min,y_min,x_max,y_max], "label": "<short description>"}]
    - Non-cropping: summarize how detected objects will be used to produce the count.
2. <think> ... </think> (only when saw the cropped image)
    - Explain how to reason step by step from the referent (or detected objects) to the final answer. 
3. <answer> ... </answer>
    - Your final answer, strictly following the format requirements in the XLRS answer format below.
Rules: 
    - Always return exactly one <answer> block, for tasks that need cropping, you can provide the bounding box of the object you are intrested, after given the cropped image, you can generate another <think> block to find the answer. 
    - For cropping tasks, also include a bounidng box in <stage_2_reasoning> block 
    - If unsure about localization, make a best guess—never say uncertain.

XLRS answer format:
Select exactly one best option. Inside <answer>, output exactly one uppercase letter from ABCD, with no other text.
<|im_end|><|im_start|>assistant
<think>  
The global satellite image shows a rural landscape with a mix of agricultural fields, patches of vegetation, and some scattered settlements. There are winding roads connecting small clusters of buildings, and the terrain includes both open farmland and forested areas. The image has a grid-like road network and varying land use patterns across different sections.  
   

This question is asking about identifying the object category within the given bounding box, therefore I need to crop the image to examine the surroundings of the mentioned target.  

Question Intent: The intent is to classify the object category based on its visual characteristics. This requires close inspection of texture, color, and shape within the specified bounding box.  

Localization Strategy: The referent object is described as being located at [25.1904, 656.9984, 59.1872, 704.2048]. In the global image, this corresponds to a small cluster of buildings situated near the bottom center, surrounded by agricultural plots. The bounding box indicates a specific area within this settlement.  

Reasoning Result:  
I need to pay attention to the reference object at  
```json
[
    {"bbox_2d": [25.1904, 656.9984, 59.1872, 704.2048], "label": "Small cluster of buildings"}
]
```
</think>
<|vision_start|><|image_pad|><|vision_end|>
~~~
