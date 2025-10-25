from __future__ import annotations
import os
import typer
from rich.console import Console
from langchain_community.document_loaders import TextLoader, DirectoryLoader, CSVLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..vectorstore import add_documents

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def ingest(path: str = typer.Option("./data", help="Path of directory (or file) to ingest")):
    docs = []
    if os.path.isdir(path):
        # Load .txt files
        txt_loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
        docs.extend(txt_loader.load())
        # Load .csv files
        csv_loader = DirectoryLoader(path, glob="**/*.csv", loader_cls=CSVLoader, show_progress=True)
        docs.extend(csv_loader.load())
        # Load .pdf files
        pdf_loader = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
        docs.extend(pdf_loader.load())
    elif os.path.isfile(path):
        lower = path.lower()
        if lower.endswith(".csv"):
            docs = CSVLoader(path).load()
        elif lower.endswith(".pdf"):
            docs = PyPDFLoader(path).load()
        else:
            docs = TextLoader(path).load()
    else:
        console.print(f"Path not found: {path}")
        raise typer.Exit(code=1)

    # Match friend's splitter more closely (smaller chunks, 0 overlap)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = splitter.split_documents(docs)
    add_documents(splits)
    console.print(f"Ingested {len(splits)} chunks into Chroma.")


if __name__ == "__main__":
    app()
