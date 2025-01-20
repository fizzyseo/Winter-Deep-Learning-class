import gradio as gr


def greet(name, is_morning):
    salutation = "Good morning" if is_morning else "Good evening"
    greeting = f"{salutation}, {name}."
    return greeting


gr.Interface(
    fn=greet,
    inputs=["text", "checkbox"],
    outputs=["text"],
).launch()