import os, shutil, time
import gradio as gr

if 'pdf-data' in os.listdir():
    print("pdf-data exists")
else:
    os.mkdir(os.path.join(os.getcwd(), 'pdf-data'))

def upload_files(files):
    file_paths = [file.name for file in files]
    return file_paths, gr.Button(value="Save", interactive=True)

def clear():
    return gr.Button(value="Save", interactive=False), gr.Markdown(visible=False)

def saveFiles(files):
    desPath = os.path.join(os.getcwd(), 'pdf-data')
    storedFiles = os.listdir(desPath)
    filesExist = []
    for file in files:
        fileName = file.name.split('/')[-1]
        if fileName not in storedFiles:
            shutil.copy(file.name, desPath)
        else:
            filesExist.append(fileName)
    
    gr.Info("Files have been uploaded!")

    if len(filesExist) > 0:
        value = "<h4 style='color:red'>Below files already exist, hence they have been ignored.</h4>\n"
        value += '<br>'.join(filesExist)
        print(value)
        return gr.Markdown(value=value, visible=True)

with gr.Blocks() as app:
    gr.Markdown("""
        <h1>llama2 and Langchain</h1>
        <p>Upload pdfs and have a conversation with the chatbot</p>
    """)
    with gr.Tab('Upload PDFs') as upload:
        file_output = gr.File(interactive=False,
                              file_count='multiple',
                              file_types=['.pdf'],
                              label="Staging")
        
        fWarn = gr.Markdown("#### Below files already exist, hence they have been ignored.")
        fWarn.visible = False

        with gr.Row():
            upload_btn = gr.UploadButton(label="Upload",
                                         file_count='multiple',
                                         file_types=['.pdf'])
            save_btn = gr.Button(value="Save", interactive=False)

        with gr.Row():
            train_btn = gr.Button(value="Train the model")
            clear_btn = gr.ClearButton(file_output)
    
    with gr.Tab('Chat') as chatbot:
        def chat(message, history):
            for i in range(len(message)):
                time.sleep(0.05)
                yield "Bot: " + message[: i+1]
        gr.ChatInterface(chat).queue()

    upload_btn.upload(upload_files, upload_btn, [file_output, save_btn])
    clear_btn.click(clear, None, [save_btn, fWarn])
    save_btn.click(saveFiles, file_output, fWarn)

app.queue().launch()