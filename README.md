# VLM-AGENT-FOR-SAM

该项目利用vlm来检测图像中的对象，并生成对应的bbox和point prompt 用来指导sam分割前景对象。

## HOW TO USE
pip install OpenAI

以及设置vlm api： set VLM_API_KEY="API"

1、vlm_bbox.py：设置合适的提示词指导vlm执行任务

2、run_one_image.py：vlm的主文件，以及包括可视化VLM的输出结果
  python run_one_image.py -i $image_path$

3、sam_segment_from_box.py：利用vlm得到的结果指导sam进行分割，可设置sam的输入
  python sam_segment_from_box.py --img "$img" \
    --bbox_json "$bbox_json" \
    --output_root "$OUTPUT_ROOT" \

4、可修改run.sh执行多图像执行

### Attention
1、run_folder.py 用于单对话框重复处理不同图像，vlm会受到之前输入的影响，导致后续效果较差。
2、为了防止期间vlm响应断了，再次起任务时不再重复执行已经处理好的文件，本文件通过bbox_json文件是否存在来检查当前图片是否已经得到处理，从而判断是否跳过该图像
