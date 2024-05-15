import gradio as gr


def generate_text():
    # 这里应当是你的文本生成逻辑
    # 为了示例，我们假设已经生成了以下字符串
    generated_text = "这是生成的文本。"
    return generated_text


# 创建一个Gradio界面
iface = gr.Interface(fn=generate_text,
                     inputs=None,
                     outputs="text",
                     title="文本生成示例",
                     description="点击按钮以生成文本。")

if __name__ == "__main__":
    iface.launch()
