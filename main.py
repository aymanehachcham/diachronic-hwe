from hwe.data.processor import DocumentProcessor

if __name__ == "__main__":
    # This is the main entry point
    docs = DocumentProcessor(json_file_path="sample_data/TheNewYorkTimes1980.xml").retrieve_context(
        query="Financial issues with the government"
    )
    for doc in docs:
        print(doc)
