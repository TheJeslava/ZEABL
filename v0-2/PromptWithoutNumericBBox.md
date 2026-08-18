# v0-2 Prompt 示例（无题面数值框）

示例问题与根目录 [v0-1PromptWithoutNumericBBox.md](../v0-1PromptWithoutNumericBBox.md) 相同。以下示例按当前 `src/eval/infer_xlrs_lite.py` 和对应运行脚本构造。

> 路由：无题面框时，stage1 使用原问题，crop 使用 stage1 输出的 `model_bbox_raw`。对应运行脚本：`run_scripts/run_balanced_50.sh`。

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
<|vision_start|><|image_pad|><|vision_end|>Why is there a cluster of barges in the middle of the river in the satellite images?

The choices are listed below:
(A) The barge might be waiting to load or unload goods at a nearby port.
(B) This river is the only navigable route, forcing all barges to pass through this section.
(C) These barges are part of the local festival and are temporarily moored in the river.
(D) This river is a popular fishing spot, attracting barges for recreational purposes.
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
<|vision_start|><|image_pad|><|vision_end|>Why is there a cluster of barges in the middle of the river in the satellite images?

The choices are listed below:
(A) The barge might be waiting to load or unload goods at a nearby port.
(B) This river is the only navigable route, forcing all barges to pass through this section.
(C) These barges are part of the local festival and are temporarily moored in the river.
(D) This river is a popular fishing spot, attracting barges for recreational purposes.
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
The global satellite image shows a large river running diagonally across the frame, with multiple barge-like structures floating in its middle section. Surrounding the river are various landmasses, some with visible industrial or construction zones, and road networks connecting different areas.  
   

Question Intent: This question is asking about the purpose or function of the cluster of barges in the middle of the river, therefore I need to crop the image to examine the surroundings of the mentioned target.

Localization Strategy: The referent object is the cluster of barges located in the middle of the river. In the global image, these appear as elongated rectangular shapes floating on the water, positioned roughly in the central part of the river channel.

Reasoning Result:  
I need to pay attention to the reference object at  
```json
[
    {"bbox_2d": [375.0, 548.0, 456.5, 619.0], "label": "cluster of barges"}
]
```
</think>
<|vision_start|><|image_pad|><|vision_end|>
~~~
