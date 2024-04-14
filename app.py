import os
import streamlit as st
import pprint
from IPython.display import Markdown, display
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.response_synthesizers import Refine, get_response_synthesizer
from llama_index.core.prompts import PromptTemplate
from llama_index.finetuning import EmbeddingAdapterFinetuneEngine
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.core.embeddings import resolve_embed_model
from llama_index.embeddings.adapter.utils import TwoLayerNN
from llama_index.llms.anthropic import Anthropic
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

documents_file = ["./uu_no_13_th_2003.pdf", "./uu_13_explained.pdf"]
pp = pprint.PrettyPrinter(indent=4)


def initialize_index(file):
    llm = Anthropic(model="claude-3-sonnet-20240229", api_key=st.secrets.anthropic.api_key, max_tokens=4096, temperature=0)
    dataset = EmbeddingQAFinetuneDataset.from_json("uu13_dataset.json")
    base_embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    adapter_model = TwoLayerNN(
        1024,
        8194,
        1024,
        bias=True,
        add_residual=True,
    )

    finetune_engine = EmbeddingAdapterFinetuneEngine(
        dataset,
        base_embed_model,
        model_output_path="model5_output_test",
        model_checkpoint_path="model5_ck",
        adapter_model=adapter_model,
        epochs=5,
        verbose=True,
        dim=1024
    )

    embed_model = finetune_engine.get_finetuned_model(
        adapter_cls=TwoLayerNN
    )
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    pc = Pinecone(api_key=st.secrets.pinecone.api_key)
    pc_index = pc.Index("uu13")
    
    vector_store = PineconeVectorStore(
        pinecone_index=pc_index
    )
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    return index

def display_prompt_dict(prompts_dict):
    for k, p in prompts_dict.items():
        text_md = f"**Prompt Key**: {k}<br>" f"**Text:** <br>"
        display(Markdown(text_md))
        print(p.get_template())
        display(Markdown("<br><br>"))

def query_index(query_text):
    qa_prompt_tmpl_str = (
        "Anda adalah bagian dari kelas pekerja.\n"
        "Perhatikan pertanyaan dengan teliti, apabila dalam pertanyaan tidak ada kaitannya dengan kelalaian pekerja, jangan berasumsi prakondisi dari pertanyaan karena adanya kelalaian pekerja.\n"
        "Perhatikan dengan teliti bahwa istilah berikut memiliki makna yang berbeda: upah, upah kerja lembur, uang pesangon, uang penghargaan masa kerja, dan uang penggantian hak.\n"
        "Karena istilah upah, upah kerja lembur, uang pesangon, uang penghargaan masa kerja, dan uang penggantian hak memiliki makna yang berbeda, maka sanksinya juga mungkin berbeda.\n"
        "Baca konteks secara teliti, karena nanti akan diajukan pertanyaan berdasarkan konteks.\n"
        "Informasi mengenai konteks berada di bawah ini.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Jawablah berdasarkan konteks.\n"
        "Jawablah dengan teliti, karena anda akan memperoleh hadiah jutaan rupiah apabila menjawab dengan tepat.\n"
        "Perhatikan dan analisis dengan teliti, apabila terdapat pernyataan matematis seperti: lebih dari, kurang dari, atau sejenisnya, jika anda ingin memperoleh hadiah milyaran rupiah\n"
        "Pertanyaan: {query_str}\n"
        "Jawaban: "
    )
    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

    refine_prompt_tmpl_str = (
        "Pertanyaan asli sebagai berikut: {query_str}\n"
        "Jawaban yang ditemukan sebagai berikut: {existing_answer}\n"
        "Perhatikan pertanyaan dengan teliti, apabila dalam pertanyaan tidak ada kaitannya dengan kelalaian pekerja, jangan berasumsi prakondisi dari pertanyaan karena adanya kelalaian pekerja.\n"
        "Perhatikan dengan teliti bahwa istilah berikut memiliki makna yang berbeda: upah, upah kerja lembur, uang pesangon, uang penghargaan masa kerja, dan uang penggantian hak.\n"
        "Karena istilah upah, upah kerja lembur, uang pesangon, uang penghargaan masa kerja, dan uang penggantian hak memiliki makna yang berbeda, maka sanksinya juga mungkin berbeda.\n"
        "Sertakan sanksi atau denda secara detail, apabila terdapat dalam konteks. Jangan ragu untuk menjawab panjang.\n"
        "Anda punya kesempatan untuk memperbaiki atau merubah jawaban menggunakan konteks di bawah ini.\n"
        "---------------------\n"
        "{context_msg}\n"
        "---------------------\n"
        "Dengan konteks baru tersebut, ubahlah jawaban awal agar lebih sesuai untuk menjawaban pertanyaan. Jika konteks tidak berguna, gunakan jawaban awal.\n"
        "Tidak perlu menyisipkan alasan dari perbaikan jawaban, cukup tampilkan jawaban terakhir.\n"
        "Apabila anda dapat menjawab dengan teliti, tanpa ada terminologi yang keliru, anda akan memperoleh hadiah jutaan rupiah.\n"
        "Perhatikan dan analisis dengan teliti, apabila terdapat pernyataan matematis seperti: lebih dari, kurang dari, atau sejenisnya, jika anda ingin memperoleh hadiah milyaran rupiah\n"
        "Jawaban Baru: "
    )
    refine_prompt_tmpl = PromptTemplate(refine_prompt_tmpl_str)
    
    _index = initialize_index(documents_file)
    
    if _index is None:
        return "Please initialize the index!"
    
    response_synthesizer = get_response_synthesizer(response_mode="refine")
    query_engine = _index.as_query_engine(
        similarity_top_k=5,
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window")
        ],
        response_synthesizer=response_synthesizer,
    )
    query_engine.update_prompts({
        "response_synthesizer:text_qa_template": qa_prompt_tmpl,
        "response_synthesizer:refine_template": refine_prompt_tmpl
    })

    summarizer = Refine(refine_template=refine_prompt_tmpl, structured_answer_filtering=True)
    
    response = query_engine.query(query_text)
    windows = [node.metadata["window"] for node in response.source_nodes]
    summarized_res = summarizer.get_response(query_text, windows)
    return summarized_res


st.title("PekerjaGPT 🚩")

text = st.text_area("Query text:", value="Secara detail, sanksi seperti apa yang dapat dikenakan apabila pengusaha tidak membayar upah lembur?")

if st.button("Submit") and text is not None:
    response = query_index(text)
    st.markdown(response)