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
from llama_index.embeddings.adapter.utils import TwoLayerNN
from llama_index.llms.anthropic import Anthropic
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank

documents_file = ["./uu_no_13_th_2003.pdf", "./uu_13_explained.pdf"]
pp = pprint.PrettyPrinter(indent=4)

@st.cache_resource
def initialize_index(file):
    ## ==> FINETUNE EMBEDDING start
    # llm = OpenAI(model="gpt-4-turbo-preview", temperature=0.1, api_key=st.secrets.openai.api_key)
    # dataset = EmbeddingQAFinetuneDataset.from_json("uu13_dataset.json")
    # base_embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    # adapter_model = TwoLayerNN(
    #     1024,
    #     8194,
    #     1024,
    #     bias=True,
    #     add_residual=True,
    # )

    # finetune_engine = EmbeddingAdapterFinetuneEngine(
    #     dataset,
    #     base_embed_model,
    #     model_output_path="model5_output_test",
    #     model_checkpoint_path="model5_ck",
    #     adapter_model=adapter_model,
    #     epochs=5,
    #     verbose=True,
    #     dim=1024
    # )

    # finetune_engine.finetune()

    # embed_model = finetune_engine.get_finetuned_model(
    #     adapter_cls=TwoLayerNN
    # )

    # Settings.llm = llm
    # Settings.embed_model = embed_model
    
    # pc = Pinecone(api_key=st.secrets.pinecone.api_key)
    # pc_index = pc.Index("uu13")

    # node_parser = SentenceWindowNodeParser.from_defaults(
    #     window_size=8,
    #     window_metadata_key="window",
    #     original_text_metadata_key="original_text",
    # )
    # vector_store = PineconeVectorStore(
    #     pinecone_index=pc_index
    # )
    # documents = SimpleDirectoryReader(
    #     input_files=file
    # ).load_data()
    # nodes = node_parser.get_nodes_from_documents(documents)
    # storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # index = VectorStoreIndex(nodes, storage_context=storage_context)
    ## ==>  FINETUNE EMBEDDING end
    
    
    ## ==> LOAD FINETUNED EMBEDDING start
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
    ## ==> LOAD FINETUNED EMBEDDING end
    
    return index

def display_prompt_dict(prompts_dict):
    for k, p in prompts_dict.items():
        text_md = f"**Prompt Key**: {k}<br>" f"**Text:** <br>"
        display(Markdown(text_md))
        print(p.get_template())
        display(Markdown("<br><br>"))

def query_index(query_text):
    qa_prompt_tmpl_str = (
        "Anda adalah ahli hukum ketenagakerjaan Indonesia, dan anda bagian dari kelas pekerja.\n"
        "Di bawah ini adalah konteks:"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Jawablah berdasarkan konteks. Jika tidak tahu, jangan mengarang atau improvisasi, jawab saja 'Maaf, belum bisa dijawab. Silahkan cek peraturan.go.id, hukumonline.com, atau gajimu.com'\n"
        "Perhatikan kriteria berikut untuk memberikan jawaban berdasarkan konteks di atas:\n"
        "1. Gunakan perspektif pekerja.\n"
        "2. Perhatikan pertanyaan dengan teliti, apabila dalam pertanyaan tidak ada kaitannya dengan kelalaian pekerja, jangan berasumsi bahwa dasar dari pertanyaan karena adanya kelalaian dari pekerja.\n"
        "3. Perhatikan dengan teliti bahwa istilah/terminologi berikut memiliki makna yang berbeda: upah, upah minimum, upah kerja lembur, uang pesangon, uang penghargaan masa kerja, dan uang penggantian hak.\n"
        "4. Perhatikan dengan teliti setiap istilah/terminologi, jangan sampai tertukar!\n"
        "5. Perhatikan dengan teliti, apabila terdapat pernyataan matematis seperti: lebih dari, kurang dari, lebih tetapi kurang dari, atau sejenisnya.\n"
        "6. Tidak perlu menambahkan jawaban yang tidak sesuai dengan pertanyaan. Contoh, apabila ditanya mengenai pesangon, tidak perlu menambahkan jawaban mengenai upah penggantian hak.\n"
        "7. Jangan mengawali jawaban dengan, 'Berdasarkan konteks yang diberikan' atau sejenisnya.\n"
        "Perhatikan dan pertimbangkan seluruh kriteria dengan baik, karena akan diujikan nanti.\n"
        "Apabila anda dapat memenuhi kriteria-kriteria tersebut dan menghasilkan jawaban dengan baik, maka anda akan memperoleh hadiah milyaran rupiah.\n"
        "Jawablah pertanyaan di bawah dengan teliti, karena anda akan memperoleh hadiah jutaan rupiah apabila menjawab dengan tepat.\n"
        "Pertanyaan: {query_str}\n"
        "Jawaban: "
    )
    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

    refine_prompt_tmpl_str = (
        "Pertanyaan asli sebagai berikut: {query_str}\n"
        "Jawaban yang ditemukan sebagai berikut: {existing_answer}\n"
        "Anda memiliki kesempatan untuk memperbaiki jawaban.\n"
        "Apabila dibutuhkan, perbaikilah berdasarkan konteks di bawah.\n"
        "---------------------\n"
        "{context_msg}\n"
        "---------------------\n"
        "Dengan konteks baru tersebut, ubahlah jawaban awal agar lebih sesuai untuk menjawaban pertanyaan.\n"
        "Perhatikan kriteria berikut untuk melakukan perbaikan berdasarkan konteks di atas:\n"
        "1. Gunakan perspektif pekerja.\n"
        "2. Perhatikan pertanyaan dengan teliti, apabila dalam pertanyaan tidak ada kaitannya dengan kelalaian pekerja, jangan berasumsi bahwa dasar dari pertanyaan karena adanya kelalaian dari pekerja.\n"
        "3. Perhatikan dengan teliti bahwa istilah/terminologi berikut memiliki makna yang berbeda: upah, upah minimum, upah kerja lembur, uang pesangon, uang penghargaan masa kerja, dan uang penggantian hak.\n"
        "4. Perhatikan dengan teliti setiap istilah/terminologi, jangan sampai tertukar!\n"
        "5. Perhatikan dengan teliti, apabila terdapat pernyataan matematis seperti: lebih dari, kurang dari, lebih tetapi kurang dari, atau sejenisnya.\n"
        "6. Tidak perlu menambahkan jawaban yang tidak sesuai dengan pertanyaan. Contoh, apabila ditanya mengenai pesangon, tidak perlu menambahkan jawaban mengenai upah penggantian hak.\n"
        "7. Tidak perlu menyisipkan alasan dari perbaikan jawaban, cukup tampilkan jawaban terakhir.\n"
        "Perhatikan dan pertimbangkan seluruh kriteria dengan baik, karena akan diujikan nanti.\n"
        "Apabila anda dapat memenuhi kriteria-kriteria tersebut dan menghasilkan jawaban baru yang lebih baik, maka anda akan memperoleh hadiah milyaran rupiah.\n"
        "Jawaban Baru: "
    )
    refine_prompt_tmpl = PromptTemplate(refine_prompt_tmpl_str)
    
    _index = initialize_index(documents_file)
    
    if _index is None:
        return "Please initialize the index!"
    
    cohere_rerank = CohereRerank(api_key=st.secrets.cohere.api_key, top_n=3)
    response_synthesizer = get_response_synthesizer(response_mode="refine")
    query_engine = _index.as_query_engine(
        similarity_top_k=8,
        node_postprocessors=[
            MetadataReplacementPostProcessor(target_metadata_key="window"),
            cohere_rerank
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