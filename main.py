from hwe.data.processor import DocumentProcessor

if __name__ == "__main__":
    # This is the main entry point
    txt = DocumentProcessor(file_path="post_process_data/compiled_docs/TheNewYorkTimes1980.json").retrieve_docs("abuse")

    print(txt)
