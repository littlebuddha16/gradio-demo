import os, shutil, time, pickle
import gradio as gr
import torch
from auto_gptq import AutoGPTQForCausalLM
from langchain.llms import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
#from pdf2image import convert_from_path
from transformers import AutoTokenizer, TextStreamer, pipeline

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
db = ""
qa_chain = ""

parentSubDirs = [item for item in os.listdir() if os.path.isdir(item)]

if 'pdf-data' not in parentSubDirs:
    os.mkdir(os.path.join(os.getcwd(), 'pdf-data'))
    os.mkdir(os.path.join(os.getcwd(), 'pdf-data', 'Unprocessed'))
    os.mkdir(os.path.join(os.getcwd(), 'pdf-data', 'Processed'))
else:
    print("pdf-data exists")

try:
    if 'embeddings' not in parentSubDirs:
        os.mkdir(os.path.join(os.getcwd(), 'embeddings'))
        
        em_instance = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large", model_kwargs={"device": DEVICE}
        )

        filePath = os.path.join(os.getcwd(), 'embeddings', 'hkunlp-instructor-large.pkl')
        with open(filePath, 'wb') as file:
            pickle.dump(em_instance, file)
    
    if 'models' not in parentSubDirs:
        os.mkdir(os.path.join(os.getcwd(), 'models'))

        modelName = "TheBloke/Llama-2-13B-chat-GPTQ"
        modelBase = "model"

        tokenizer = AutoTokenizer.from_pretrained(modelName, use_fast=True)

        model_instance = AutoGPTQForCausalLM.from_quantized(
            modelName,
            revision="gptq-4bit-128g-actorder_True",
            model_basename=modelBase,
            use_safetensors=True,
            trust_remote_code=True,
            inject_fused_attention=False,
            device=DEVICE,
            quantize_config=None,
        )

        modelPath = os.path.join(os.getcwd(), 'models', 'TheBloke-Llama-2-13B-chat-GPTQ.pkl')

        with open(modelPath, 'wb') as file:
            pickle.dump(model_instance, file)

finally:
    print("embeddings exist")
    print("models exist")
    modelName = "TheBloke/Llama-2-13B-chat-GPTQ"
    tokenizer = AutoTokenizer.from_pretrained(modelName, use_fast=True)
    embeddingsFile = os.path.join(os.getcwd(), 'embeddings', 'hkunlp-instructor-large.pkl')
    modelFile = os.path.join(os.getcwd(), 'models', 'TheBloke-Llama-2-13B-chat-GPTQ.pkl')
    
    with open(embeddingsFile, 'rb') as file:
        embeddings = pickle.load(file)
        print(f"embeddings file loaded: {type(embeddings)}")
    
    with open(modelFile, 'rb') as file:
        model = pickle.load(file)
        print(f"model loaded: {type(model)}")


DEFAULT_SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
""".strip()

def generate_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return f"""
[INST] <<SYS>>
{system_prompt}
<</SYS>>

{prompt} [/INST]
""".strip()

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0,
    top_p=0.95,
    repetition_penalty=1.15,
    streamer=streamer,
)

llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})

SYSTEM_PROMPT = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."

template = generate_prompt(
    """
{context}

Question: {question}
""",
    system_prompt=SYSTEM_PROMPT,
)

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# print(embeddings)
def upload_files(files):
    file_paths = [file.name for file in files]
    return file_paths, gr.Button(value="Save", interactive=True), "Uploaded files"

def clear():
    return gr.Button(value="Save", interactive=False), gr.Markdown(visible=False), "Idle"

def saveFiles(files):
    desPath = os.path.join(os.getcwd(), 'pdf-data', 'Unprocessed')
    processedPath = os.path.join(os.getcwd(), 'pdf-data', 'Processed')
    storedFiles = os.listdir(processedPath)
    
    filesExist = []
    for file in files:
        fileName = file.name.split('\\')[-1]
        if fileName not in storedFiles:
            shutil.copy(file.name, desPath)
            print(f"fileCopied: {fileName}")
        else:
            filesExist.append(fileName)
    
    gr.Info("Files have been uploaded!")

    if len(filesExist) > 0:
        value = "<h4 style='color:red'>Below files already exist, hence they have been ignored.</h4>\n"
        value += '<br>'.join(filesExist)
        return (gr.Markdown(value=value, visible=True), "Files have been saved")
    else:
        return ("", "Files have been saved")

def trainModel():
    sourcePath = os.path.join(os.getcwd(), 'pdf-data', 'Unprocessed')
    desPath = os.path.join(os.getcwd(), 'pdf-data', 'Processed')
    loader = PyPDFDirectoryLoader(sourcePath)
    docs = loader.load()
    for file in os.listdir(sourcePath):
        shutil.move(os.path.join(sourcePath, file), desPath)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=80)
    splitText = text_splitter.split_documents(docs)
    db = Chroma.from_documents(splitText, embeddings, persist_directory="db")
    print(f"db after: {type(db)}")
    global qa_chain
    return "Embeddings saved to db"


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

        status_box = gr.Textbox(label="Status", placeholder="Idle", interactivity=False)

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
            qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )
            response = qa_chain(message)
            for i in range(len(response)):
                time.sleep(0.05)
                yield "Bot: " + response[: i+1]
        gr.ChatInterface(chat).queue()

    upload_btn.upload(upload_files, upload_btn, [file_output, save_btn, status_box])
    clear_btn.click(clear, None, [save_btn, fWarn, status_box])
    save_btn.click(saveFiles, file_output, [fWarn, status_box])
    train_btn.click(trainModel, None, status_box)

app.queue().launch()