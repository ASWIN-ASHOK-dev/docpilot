import typer
from chat import askai
from scrape import scrape_url, scrape_sitemap, scrape_site
from embed import embed_texts

app = typer.Typer()

@app.command()
def ingest(source: str):
    if source.endswith("sitemap.xml"):
        texts = scrape_sitemap(source)
    elif source.startswith("http"):
        texts = scrape_site(source)
    else:
        print("Unsupported source")
        return
    embed_texts(texts, source=source)

@app.command()
def ask(question: str):
    """Ask a question against the ingested docs."""
    print(f"Question: {question}")
    askai(question)

if __name__ == "__main__":
    app()