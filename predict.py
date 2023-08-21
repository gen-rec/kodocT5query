import argparse
import json
from pathlib import Path
from typing import Literal

import torch
from pytorch_lightning import seed_everything
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5TokenizerFast

from src.utils import load_map


@torch.no_grad()
def generate_query(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    document: str,
    device: torch.device,
    strategy: Literal["greedy", "beam", "top_k"],
    num_queries: int,
    batch_size: int,
    query_max_length: int = 64,
    max_tries: int = 256,
    **kwargs,
) -> list[str]:
    """
    Generate queries for a document

    Three methods are supported:

    * Greedy: Generate a single query using greedy decoding
    * Beam: Generate multiple queries using beam search
    * Top-k: Generate multiple queries using top-k sampling

    For greedy decoding, ``num_queries`` must be 1.

    For beam search, ``num_queries`` is the number of beams to use.

    For top-k sampling, ``num_queries`` is the number of queries to generate.
    The top-k sampling is repeated until ``num_queries`` unique queries are generated.

    Args:
        model (T5ForConditionalGeneration): T5 model
        tokenizer (T5Tokenizer): T5 tokenizer
        document (str): Document to generate queries for
        device (torch.device): Device to run model on
        strategy (Literal["greedy", "beam", "top_k"]): Strategy to use for generating queries
        num_queries (int): Number of queries to generate
        batch_size (int): Batch size for top-k sampling
        query_max_length (int, optional): Max length of query. Defaults to 64.
        max_tries (int, optional): Max number of tries for top-k sampling. Defaults to 200.
        kwargs: Keyword arguments to pass to the model

    Raises:
        ValueError: If strategy is not "greedy", "beam", or "top_k"
        ValueError: If strategy is "greedy" and num_queries is not 1
        ValueError: If num_queries is less than 1

    Returns:
        list[str]: List of queries
    """
    if strategy == "greedy" and num_queries != 1:
        raise ValueError("Greedy strategy only supports 1 query")
    if num_queries < 1:
        raise ValueError("Number of queries must be greater than 0")

    if strategy == "greedy":
        inputs = tokenizer(document, truncation=True, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=query_max_length, **kwargs)
        result = [tokenizer.decode(outputs[0], skip_special_tokens=True)]
    elif strategy == "beam":
        inputs = tokenizer(document, truncation=True, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=query_max_length,
            num_beams=num_queries,
            num_return_sequences=num_queries,
            **kwargs,
        )
        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    elif strategy == "top_k":
        # Repeat top_k sampling until we have num_queries unique queries
        result = []

        inputs = tokenizer([document] * batch_size, truncation=True, return_tensors="pt").to(device)

        queries_progress = tqdm(total=num_queries, desc="Generating queries", position=1, ncols=120, leave=False)
        total_tries = 0
        while len(result) < num_queries:
            outputs = model.generate(**inputs, do_sample=True, max_new_tokens=query_max_length, top_k=10, **kwargs)
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Filter out duplicates
            for decoded_output in decoded_outputs:
                if decoded_output not in result:
                    result.append(decoded_output)
                    queries_progress.update(1)

                total_tries += 1
                queries_progress.set_postfix({"tries": total_tries})

                if len(result) >= num_queries:
                    break

                if total_tries >= max_tries:
                    print(f"Max tries ({max_tries}) reached. Generated {len(result)} queries.")
                    break

        queries_progress.close()
    else:
        raise ValueError("Invalid strategy")

    return result


def generate_queries_for_collection(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    collection: dict[str, str],
    output_path: str | Path,
    device: torch.device,
    strategy: Literal["greedy", "beam", "top_k"],
    num_queries: int,
    batch_size: int = 8,
    query_max_length: int = 64,
    **kwargs,
):
    """
    Generate queries for a collection of documents

    For more information, see ``generate_query``

    Args:
        model (T5ForConditionalGeneration): T5 model
        tokenizer (T5Tokenizer): T5 tokenizer
        collection (dict[str, str]): Collection of documents
        output_path (str | Path): Path to save the generated queries and expanded collection
        device (torch.device): Device to run model on
        strategy (Literal["greedy", "beam", "top_k"]): Strategy to use for generating queries
        num_queries (int): Number of queries to generate
        batch_size (int, optional): Batch size for top-k sampling. Defaults to 8.
        query_max_length (int, optional): Max length of query. Defaults to 64.
        kwargs: Keyword arguments to pass to the model

    Raises:
        ValueError: If strategy is not "greedy", "beam", or "top_k"
        ValueError: If strategy is "greedy" and num_queries is not 1
        ValueError: If num_queries is less than 1

    Returns:
        list[str]: List of queries
    """
    generated_queries = {}  # docid -> list[query]
    collection_expanded = {}  # docid -> document + query

    try:
        for docid, doc in tqdm(collection.items(), desc="Expanding documents", position=0, ncols=120):
            generated = generate_query(
                model=model,
                tokenizer=tokenizer,
                document=doc,
                device=device,
                strategy=strategy,
                num_queries=num_queries,
                batch_size=batch_size,
                query_max_length=query_max_length,
                **kwargs,
            )
            generated_queries[docid] = generated
            collection_expanded[docid] = " ".join([doc] + generated)
    except KeyboardInterrupt:
        print("Interrupted. Saving generated queries and expanded collection...")

    # Save
    with open(output_path / "generated_queries.json", "w", encoding="utf-8") as f:
        json.dump(generated_queries, f, indent=1, ensure_ascii=False)

    with open(output_path / "collection_expanded.tsv", "w", encoding="utf-8") as f:
        for docid, doc in collection_expanded.items():
            f.write(f"{docid}||{doc}\n")

    print(f"Saved generated queries to {output_path / 'generated_queries.json'}")
    print(f"Saved expanded collection to {output_path / 'collection_expanded.tsv'}")


def predict(
    model_path: str,
    tokenizer_path: str | None = None,
    num_queries: int = 10,
    batch_size: int = 8,
    strategy: Literal["greedy", "beam", "top_k"] = "top_k",
    document_str: str | None = None,
    document_path: str | None = None,
    output_path: str | Path | None = None,
    device: torch.device = torch.device("cpu"),
    seed: int | None = None,
):
    """
    Generate queries for a document

    If a single document string(``document_str``) is provided, the queries will be generated for that document.
    The queries will be printed to stdout.

    If a document collection file(``document_path``) is provided,
    the queries will be generated for each document in the collection.

    Two files are saved:

    * (``collection_expanded.tsv``) A document collection file with the queries appended to each document
    * (``generated_queries.json``) A json file mapping docid to generated queries

    Only one of ``document_str`` and ``document_path`` can be provided. Providing both will raise a ValueError.

    Args:
        model_path (str): Path to the model (Huggingface model hub or local path)
        tokenizer_path (str, optional): Path to the tokenizer (Huggingface model hub or local path). Defaults to None.
            If None, the tokenizer will be loaded from the model_path.
        num_queries (int, optional): Number of queries to generate. Defaults to 10.
        batch_size (int, optional): Batch size for top-k sampling. Defaults to 8.
        strategy (Literal["greedy", "beam", "top_k"], optional): Strategy to use for generating queries.
            Defaults to "top_k".

            * Greedy: Generate a single query using greedy decoding
            * Beam: Generate multiple queries using beam search
            * Top-k: Generate multiple queries using top-k sampling
        document_str (str, optional): Document to generate queries for. Defaults to None.
        document_path (str, optional): Path to the document to generate queries for. Defaults to None.
        output_path (str, optional): If document_path is provided, the expanded collection and generated queries will
            be saved to this path. Defaults to None.

            If None, the expanded collection and generated queries will be
            saved to the same directory as the document_path.
        device (torch.device, optional): Device to run model on. Defaults to torch.device("cpu").
        seed (int, optional): Random seed for reproducibility of top-k sampling. Defaults to 42.

    Raises:
        ValueError: If neither document_str nor document_path is provided
        ValueError: If both document_str and document_path is provided

    Returns:

    """
    if document_str is None and document_path is None:
        raise ValueError("Either document or document_path must be provided.")

    if document_str is not None and document_path is not None:
        raise ValueError("Only one of document or document_path must be provided.")

    if tokenizer_path is None:
        tokenizer_path = model_path

    if seed is not None:
        seed_everything(seed)

    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path, model_max_length=512)

    if document_str is None:
        # Load document from file
        document_path: str

        print(f"Loading document from {document_path}")

        assert document_path.endswith(".tsv"), "Document must be a .tsv file"
        output_path = Path(output_path) if output_path is not None else Path(document_path).parent
        collection = load_map(document_path, "||")

        generate_queries_for_collection(
            model=model,
            tokenizer=tokenizer,
            collection=collection,
            output_path=output_path,
            device=device,
            strategy=strategy,
            batch_size=batch_size,
            num_queries=num_queries,
        )

    else:
        # Use document provided
        print(f"Generating queries for document:\n{document_str}")
        generated_query = generate_query(
            model=model,
            tokenizer=tokenizer,
            document=document_str,
            device=device,
            strategy=strategy,
            num_queries=num_queries,
            batch_size=batch_size,
        )
        print("\nGenerated queries:")
        for i, query in enumerate(generated_query, start=1):
            print(f"{i}. {query}")


def _main():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model (Huggingface model hub or local path)"
    )
    arg_parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to the tokenizer (Huggingface model hub or local path). Defaults to None. "
        "If None, the tokenizer will be loaded from the model_path.",
    )
    arg_parser.add_argument(
        "--num_queries", type=int, default=10, help="Number of queries to generate for each document. Defaults to 10."
    )
    arg_parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for top-k sampling. Defaults to 8."
    )
    arg_parser.add_argument(
        "--strategy",
        type=str,
        default="top_k",
        choices=["greedy", "beam", "top_k"],
        help="Strategy to use for generating queries. Defaults to top_k.",
    )
    arg_parser.add_argument(
        "--document_str", type=str, default=None, help="Document to generate queries for. Defaults to None."
    )
    arg_parser.add_argument(
        "--document_path",
        type=str,
        default=None,
        help="Path to the document to generate queries for. Defaults to None.",
    )
    arg_parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="If document_path is provided, the expanded collection and generated queries will be saved to this path. "
        "Defaults to None. If None, the expanded collection and generated queries will be saved to the same "
        "directory as the document_path.",
    )
    arg_parser.add_argument(
        "--device", type=torch.device, default=torch.device("cpu"), help="Device to run model on. Defaults to cpu."
    )
    arg_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility of top-k sampling. Defaults to 42."
    )

    args = arg_parser.parse_args()

    predict(**vars(args))


if __name__ == "__main__":
    _main()
