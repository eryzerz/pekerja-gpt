import json
import os
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader
from llama_index.finetuning import generate_qa_embedding_pairs
from llama_index.core.node_parser import SentenceWindowNodeParser

files = ["./uu_no_13_th_2003.pdf", "./uu_13_explained.pdf"]
os.environ["OPENAI_API_KEY"] = "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"

def main():

  def load_corpus(files, verbose=False):
      if verbose:
          print(f"Loading files {files}")

      reader = SimpleDirectoryReader(input_files=files)
      docs = reader.load_data()
      if verbose:
          print(f"Loaded {len(docs)} docs")

      parser = SentenceWindowNodeParser.from_defaults(
        window_size=8,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
      )
      nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

      if verbose:
          print(f"Parsed {len(nodes)} nodes")

      return nodes

  train_nodes = load_corpus(files=files, verbose=True)

  qa_generate_prompt_tmpl = """\
      Konteks dari informasi berada di bawah ini.

      ---------------------
      {context_str}
      ---------------------

      Menggunakan konteks yang diberikan yang berasal dari Undang-undang Nomor 13 Tahun 2003 tentang Ketenagakerjaan, \
      buat pertanyaan berdasarkan kondisi di bawah.

      Anda adalah seorang Ahli Hukum Ketenagakerjaan Indonesia, tugas anda adalah membuat \
      {num_questions_per_chunk} pertanyaan untuk kuis/ujian mendatang. 
      Pertanyaan yang dihasilkan wajib mewakili pertanyaan para pekerja Indonesia pada umumnya mengenai peraturan ketenagakerjaan berdasarkan konteks \
      dan bervariasi mecakup seluruh bagian dokumen, seperti mengenai hak-hak pekerja atau pengusaha, kewajiban pekerja atau pengusaha, sanksi bagi pekerja atau pengusaha yang melanggar peraturan, dan denda bagi pekerja atau pengusaha yang melanggar peraturan.\ 
      Perhatikan beragam definisi umum, untuk menghindari kekeliruan dalam membuat pertanyaan atau menyusun jawaban. Batasi pertanyaan hanya \
      pada cakupan konteks yang diberikan."
    """

  train_dataset = generate_qa_embedding_pairs(
    llm=OpenAI(model="gpt-4-turbo-preview"),
    nodes=train_nodes,
    qa_generate_prompt_tmpl=qa_generate_prompt_tmpl,
    num_questions_per_chunk=12
  )

  train_dataset.save_json("uu13_dataset_v2.json")

if __name__ == "__main__":
    main()
