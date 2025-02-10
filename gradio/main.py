import gradio as gr

# 定義主頁面
with gr.Blocks() as demo:
    gr.Markdown("# 主頁面")
    name = gr.Textbox(label="Name")
    # 在此添加更多的 Gradio 元件和功能

# 定義設定頁面
with demo.route("Settings", "/settings"):
    gr.Markdown("# 設定頁面")
    num = gr.Number(label="Number")
    # 在此添加更多的 Gradio 元件和功能

demo.launch()
