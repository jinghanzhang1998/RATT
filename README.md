# RATT: A Thought Structure for Coherent and Correct LLM Reasoning


## Introduction

Welcome to the GitHub repository for the paper ["RATT: A Thought Structure for Coherent and Correct LLM Reasoning."](https://arxiv.org/abs/2406.02746) 

Large Language Models (LLMs) gain substantial reasoning and decision-making capabilities from thought structures. However, existing methods such as Tree of Thought and Retrieval Augmented Thoughts often fall short in complex tasks due to the limitations of insufficient local retrieval of factual knowledge and inadequate global selection of strategies. These limitations make it challenging for these methods to balance factual accuracy and comprehensive logical optimization effectively. To address these limitations, we introduce the Retrieval Augmented Thought Tree (RATT), a novel thought structure that considers both overall logical soundness and factual correctness at each step of the thinking process. Specifically, at every point of a thought branch, RATT performs planning and lookahead to explore and evaluate multiple potential reasoning steps, and integrate the fact-checking ability of Retrieval-Augmented Generation (RAG) with LLM's ability to assess overall strategy. Through this combination of factual knowledge and strategic feasibility, the RATT adjusts and integrates the thought tree structure to search for the most promising branches within the search space. This thought structure significantly enhances the model's coherence in logical inference and efficiency in decision-making, and thus increases the limit of the capacity of LLM to generate reliable inferences and decisions based on thought structures. A broad range of experiments on different types of tasks showcases that the RATT structure significantly outperforms existing methods in factual correctness and logical coherence.

## Prerequisites

* If you need to run `example_creative_writing.py`, the Python packages can be installed via `pip install -r requirements.txt`.
* If you need to run `example_creative_writing_offline.ipynb`, please install the dependencies directly using the commands provided in the .ipynb file.

### Please obtain the following part or all API keys according to your needs.

* `OpenAI_API_key` is required If you need to access OpenAI's language models or embedding models (such as text-embedding-ada-002). You can get it from [OpenAI](https://beta.openai.com/signup/). 
```
os.environ["OPENAI_API_KEY"] = "" # Your Openai API key
```

* You can also use the Huggingface API key for the language model. You can get it from [Huggingface](https://huggingface.co/join).

* You also need to prepare the `GOOGLE_API_KEY` for Google Search API. You can get it from [Google Cloud Platform](https://cloud.google.com/docs/authentication/getting-started).
```
os.environ["GOOGLE_API_KEY"] = "" # Your Google API Key
```

* If you need to customize your Google Search API, for example, to restrict your searches to a specific website (such as Wikipedia), you will need a `GOOGLE_CSE_ID`. You can get it from the [Google Programmable Search Engine](https://developers.google.com/custom-search/v1/overview).
```
os.environ["GOOGLE_CSE_ID"] = "" # Your Google CSE ID
```



## Getting Started

We have released two versions of RATT, each corresponding to a RAG library based on web content and local documents.

The `example_creative_writing_offline.ipynb` notebook demonstrates an example of generating creative text by using a local file `llama2.pdf` as the library for retrieval.

The `example_creative_writing.py` script demonstrates an example of generating creative text by using English Wikipedia page content from Google Search as the library for retrieval.


## Citation
```
@misc{zhang2024rattthoughtstructurecoherent,
      title={RATT: A Thought Structure for Coherent and Correct LLM Reasoning}, 
      author={Jinghan Zhang and Xiting Wang and Weijieying Ren and Lu Jiang and Dongjie Wang and Kunpeng Liu},
      year={2024},
      eprint={2406.02746},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.02746}, 
}
```
