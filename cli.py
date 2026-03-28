import ollama
import typer
from typer import models
from chat import askai
from scrape import scrape_url, scrape_sitemap, scrape_site
from embed import embed_texts
import ollama
import store

# Initialize config at startup
store.init_config()

app = typer.Typer()


@app.command()
def ingest(
    source: str,
    max_pages: int = typer.Option(20, "--max-pages", "-p", help="Max pages to crawl for websites."),
    workers: int = typer.Option(16, "--workers", "-w", help="Concurrent workers for scraping."),
):
    if source.endswith("sitemap.xml"):
        texts = scrape_sitemap(source, max_workers=workers)
    elif source.startswith("http"):
        texts = scrape_site(source, max_pages=max_pages, max_workers=workers)
    else:
        print("Unsupported source")
        return
    embed_texts(texts, source=source)
    print("Ingestion complete.")

@app.command()
def ask(question: str):
    """Ask a question against the ingested docs."""
    print(f"Question: {question}")
    askai(question)
    print(f"\n {question} --> Done. \n")

@app.command()
def model(action: str, model_name = typer.Argument(None, help="Name of the model to set")):
    models = ollama.list()
    config = store.load_config()
    """List available models or set a default model."""
    if action == "list":
        for i in models.models:
            print(i.model)  
    elif action == "set":
        if model_name in [m.model for m in models.models]:
            config["default_model"] = model_name
            store.save_config(config)
            print(f"Model set to: {model_name}")
        else:
            print(f"Model '{model_name}' not found. Use 'list' to see available models.")
    elif action == "setembed":
        if model_name in [m.model for m in models.models]:
            config["default_embed_model"] = model_name
            store.save_config(config)
            print(f"Embed model set to: {model_name}")
        else:
            print(f"Model '{model_name}' not found. Use 'list' to see available models.")
    else:
        print("Unsupported action. Use 'list' to see available models.")



if __name__ == "__main__":
    app()