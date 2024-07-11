from openai import OpenAI
from langchain.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import AsyncHtmlLoader
from datetime import datetime
from multiprocessing import Process, Queue
from difflib import unified_diff
from IPython.display import display, HTML
import openai
import argparse
import pickle
import numpy as np
import faiss
from langchain.embeddings import OpenAIEmbeddings

from human_eval.data import read_problems
from datasets import load_dataset
import tiktoken
import os

api_key = "" # Your API key
openai_client = OpenAI(api_key=api_key)
client = openai.OpenAI(api_key=api_key)
# Basic Tool Functions
os.environ["GOOGLE_CSE_ID"] = "" # Your Google CSE ID
os.environ["GOOGLE_API_KEY"] = "" # Your Google API Key


chatgpt_system_prompt = f'''
You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.
Knowledge cutoff: 2023-04
Current date: {datetime.now().strftime('%Y-%m-%d')}
'''

library = load_dataset("codeparrot/github-jupyter")

# Load the library
docs_file = os.path.join('library', 'processed_docs.pkl')
embedding_file = os.path.join('library', 'embedded_docs.pkl')
index_file = os.path.join('library', 'index_file')

with open(docs_file, 'rb') as file:
    docs = pickle.load(file)
print("Docs have been loaded from file.")

with open(embedding_file, 'rb') as file:
    embedded_docs = pickle.load(file)
print("Embeddings have been loaded from file.")

index = faiss.read_index(index_file)
print("FAISS index has been loaded from file.")


def search_documents(query, embedding_model="text-embedding-ada-002", k=3):

    embeddings = OpenAIEmbeddings(api_key="", model=embedding_model)

    query_embedding = embeddings.embed_documents([query])[0]

    D, I = index.search(np.array([query_embedding]), k)

    results = []
    for i in range(k):
        doc_index = I[0][i]
        results.append({
            "document": docs[doc_index].page_content,
            "score": D[0][i]
        })

    return results




def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def chunk_text_by_sentence(text, chunk_size=2048):
    """Chunk the $text into sentences with less than 2k tokens."""
    sentences = text.split('. ')
    chunked_text = []
    curr_chunk = []
    for sentence in sentences:
        if num_tokens_from_string(". ".join(curr_chunk)) + num_tokens_from_string(sentence) + 2 <= chunk_size:
            curr_chunk.append(sentence)
        else:
            chunked_text.append(". ".join(curr_chunk))
            curr_chunk = [sentence]
    if curr_chunk:
        chunked_text.append(". ".join(curr_chunk))
    return chunked_text[0]

def chunk_text_front(text, chunk_size = 2048):
    '''
    get the first `trunk_size` token of text
    '''
    chunked_text = ""
    tokens = num_tokens_from_string(text)
    if tokens < chunk_size:
        return text
    else:
        ratio = float(chunk_size) / tokens
        char_num = int(len(text) * ratio)
        return text[:char_num]

def chunk_texts(text, chunk_size = 2048):
    '''
    trunk the text into n parts, return a list of text
    [text, text, text]
    '''
    tokens = num_tokens_from_string(text)
    if tokens < chunk_size:
        return [text]
    else:
        texts = []
        n = int(tokens/chunk_size) + 1
        part_length = len(text) // n
        extra = len(text) % n
        parts = []
        start = 0

        for i in range(n):
            end = start + part_length + (1 if i < extra else 0)
            parts.append(text[start:end])
            start = end
        return parts

def generate_code(query, code_summary):
    prompt_template = '''
IMPORTANT:
Based on the provided query and code summary, generate the corresponding Python code. 
Ensure the code is well-structured and directly addresses the problem specified in the query.
DO NOT include any explanations, introductions, or additional content in the output.
PROVIDE ONLY THE Python CODE
'''

    # Combine the query and code summary into the final prompt
    final_prompt = f"Query:\n{query}\n\nCode Summary:\n{code_summary}\n\n{prompt_template}"

    # Generate the Python code using the OpenAI API
    generated_code = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": chatgpt_system_prompt
            },
            {
                "role": "user",
                "content": final_prompt
            }
        ],
        temperature=1.0
    ).choices[0].message.content.strip()

    return generated_code

def get_draft_tot_inital(question, num_agents=3):

    draft_prompt = '''
IMPORTANT:
Try to answer this question/instruction with step-by-step thoughts and make the answer more structural.
Use `\n\n` to split the answer into several paragraphs.
Just respond to the instruction directly. DO NOT add additional explanations or introducement in the answer unless you are asked to.
'''

    refine_prompt = '''
Now compare all agent's answer with each other and learn. Then generate a better answer based on these answers. 
'''
    polish_prompt = '''
Try to sumarize this content with step-by-step thoughts and make the answer more structural. Reduce the redundancy of comparison and reasoning, make the answer more concise.
Use `\n\n` to split the answer into several paragraphs.
IMPORTANT:Just respond to the instruction directly. DO NOT add additional explanations or introducement in the answer unless you are asked to.
'''

    agents_drafts = []

    for i in range(num_agents):
        draft = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": chatgpt_system_prompt
                },
                {
                    "role": "user",
                    "content": question + draft_prompt
                }
            ],
            temperature=1.0
        ).choices[0].message.content


        agents_drafts.append(f"Agent{i+1}: {draft}")

        print(f"{datetime.now()} [INFO] Round 1, Agent{i + 1}/{num_agents} retrieving draft...")


    agents_input = '\n\n'.join(agents_drafts) + '\n\n' + refine_prompt

    # 生成整合答案
    final_draft_raw = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": chatgpt_system_prompt
            },
            {
                "role": "user",
                "content": agents_input
            }
        ],
        temperature=1.0
    ).choices[0].message.content

    print(f"{datetime.now()} [INFO] Round 1, retrieving integrated draft...")

    final_draft = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": chatgpt_system_prompt
            },
            {
                "role": "user",
                "content": final_draft_raw + polish_prompt
            }
        ],
        temperature=1.0
    ).choices[0].message.content

    print(f"{datetime.now()} [INFO] Round 1, removing redundant reasoning...")


    return final_draft

def get_draft_tot(question, previous_answer, num_agents=3):
    draft_prompt = f'''
Base your response on the following question and previous answer. Provide a more comprehensive answer building on the previous answer with step-by-step thoughts and make the answer more structural.
Question: {question}
Previous Answer: {previous_answer}
IMPORTANT:
Answer the full question with step-by-step thoughts and make the answer more structural.
Use `\n\n` to split the answer into several paragraphs.
Just respond to the instruction directly. DO NOT add additional explanations or introducement in the answer unless you are asked to.
'''

    refine_prompt = '''
let's consider all responses comprehensively, draw advantage of each response, and integrate them to give a coherent and logical response.
'''

    polish_prompt = '''
Try to summarize this content with step-by-step thoughts and make the answer more structural. Reduce the redundancy of comparison and reasoning, make the answer more concise.
Use `\n\n` to split the answer into several paragraphs.
IMPORTANT: Just respond to the instruction directly. DO NOT add additional explanations or introducement in the answer unless you are asked to.
'''

    agents_drafts = []

    for i in range(num_agents):
        draft = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": chatgpt_system_prompt
                },
                {
                    "role": "user",
                    "content": draft_prompt
                }
            ],
            temperature=1.0
        ).choices[0].message.content

        agents_drafts.append(f"Agent{i+1}: {draft}")

        print(f"{datetime.now()} [INFO] Round 1, Agent{i + 1}/{num_agents} retrieving draft...")

    agents_input = '\n\n'.join(agents_drafts) + '\n\n' + refine_prompt

    final_draft_raw = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": chatgpt_system_prompt
            },
            {
                "role": "user",
                "content": agents_input
            }
        ],
        temperature=1.0
    ).choices[0].message.content

    print(f"{datetime.now()} [INFO] Round 1, fetching integrated draft...")

    final_draft = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": chatgpt_system_prompt
            },
            {
                "role": "user",
                "content": final_draft_raw + polish_prompt
            }
        ],
        temperature=1.0
    ).choices[0].message.content

    print(f"{datetime.now()} [INFO] Round 1, removing redundant reasoning...")

    return final_draft


def get_draft_cot(question):
    # Getting the draft answer
    draft_prompt = '''
IMPORTANT:
Try to answer this question/instruction with step-by-step thoughts and make the answer more structural.
Use `\n\n` to split the answer into several paragraphs.
Just respond to the instruction directly. DO NOT add additional explanations or introducement in the answer unless you are asked to.
'''
    draft = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": chatgpt_system_prompt
            },
            {
                "role": "user",
                "content": f"{question}" + draft_prompt
            }
        ],
        temperature=1.0
    ).choices[0].message.content
    return draft

def get_checking_draft_cot(question, draft):
    previous_draft = draft
    # Getting the draft answer
    draft_prompt = f'''
Now according to this integrated response:

{previous_draft}

Let's provide a step-by-step review to identify any potential logical or factual errors, with a focus on any code that might cause errors.
IMPORTANT:

If no errors are found, simply state "No errors found".
Make the answer more structural, using `\\n\\n` to split the answer into several paragraphs.
JUST OUTPUT THE REVIEW DIRECTLY. DO NOT add additional content in the answer unless you are asked to.
'''
    revised_draft = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": chatgpt_system_prompt
            },
            {
                "role": "user",
                "content": f"##Question: {question}\n\n##Draft: {previous_draft}\n\n##Instruction: {draft_prompt}"
            }
        ],
        temperature=1.0
    ).choices[0].message.content.strip()
    return revised_draft

def get_revise_draft_cot(question, draft, checking_instruction):
    previous_draft = draft
    draft_prompt = f'''
According to the integrated response:

{previous_draft}

Let's optimize it for logical coherence, identifying and correcting any bugs or errors. Ensure the code is well-structured and free from unnecessary parts. 
Here are the instructions for the revision:

{checking_instruction}

When providing revision guidance, make changes according to the instructions if you believe they are correct. If the instructions themselves are incorrect, please ignore them.

IMPORTANT:
OUTPUT ONLY THE IMPROVED CODE. DO NOT add additional explanations or introductions in the answer.
'''
    revised_draft = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": chatgpt_system_prompt
            },
            {
                "role": "user",
                "content": f"##Question: {question}\n\n##Draft: {previous_draft}\n\n##Instruction: {draft_prompt}"
            }
        ],
        temperature=1.0
    ).choices[0].message.content.strip()
    return revised_draft

def get_query(question, answer):
    query_prompt = '''
I want to verify the code correctness of the given question, especially protential bug in the answer.
Please summarize the content with the corresponding question.
This summarization will be used as a query to search in a local code database.
The query should be short but need to be specific to promise I can find related knowledge or pages.
You can use programming-specific keywords and syntax to make the query more accurate for finding related code snippets.
**IMPORTANT**
Just output the query directly. DO NOT add additional explanations or introducement in the answer unless you are asked to.
'''
    query = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": chatgpt_system_prompt
            },
            {
                "role": "user",
                "content": f"##Question: {question}\n\n##Content: {answer}\n\n##Instruction: {query_prompt}"
            }
        ],
        temperature=1.0
    ).choices[0].message.content
    return query


def integrate_agents_answers(question, all_agent_results):
    refine_prompt = '''
Let's consider all responses comprehensively, draw advantage of each response, and integrate them to give a coherent and logical response.
Identify and correct any errors, and remove unnecessary parts that are not relevant to the goal of the question.
**IMPORTANT**
Only output the final integrated code. DO NOT add any explanations or additional text.
    '''

    agents_input = '\n\n'.join(all_agent_results) + '\n\n' + refine_prompt

    final_draft_raw = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": chatgpt_system_prompt
            },
            {
                "role": "user",
                "content": f"##Question: {question}\n\n{agents_input}"
            }
        ],
        temperature=1.0
    ).choices[0].message.content

    return final_draft_raw

def get_revise_answer(question, answer, content):
    revise_prompt = '''
I want to revise the answer according to retrieved related text of the question in the code database.
You need to check whether the answer is correct.
If you find some errors in the answer, revise the answer to make it better.
If you find some unnecessary details that are not relevant to the question's goal, remove the redundant parts and ensure the code flows smoothly.
If you find the answer is right and does not need additional modifications, just output the original answer directly.
**IMPORTANT**
Output ONLY THE REVISED CODE DIRECTLY. DO NOT add additional explanations or announcements in the revised code unless you are asked to.
'''
    revised_answer = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": chatgpt_system_prompt
            },
            {
                "role": "user",
                "content": f"##Existing Text in Wiki Web: {content}\n\n##Question: {question}\n\n##Answer: {answer}\n\n##Instruction: {revise_prompt}"
            }
        ],
        temperature=1.0
    ).choices[0].message.content
    return revised_answer

def get_query_wrapper(q, question, answer):
    result = get_query(question, answer)
    q.put(result)

def get_content_wrapper(q, query):
    result = search_documents(query)
    q.put(result)

def get_revise_answer_wrapper(q, question, answer, content):
    result = get_revise_answer(question, answer, content)
    q.put(result)

def run_with_timeout(func, timeout, *args, **kwargs):
    q = Queue()  # Create a Queue object for inter-process communication
    # Create a process to execute the given function, passing the Queue and other *args, **kwargs as arguments
    p = Process(target=func, args=(q, *args), kwargs=kwargs)
    p.start()

    p.join(timeout)
    if p.is_alive():
        print(f"{datetime.now()} [INFO] Function {str(func)} execution has timed out ({timeout}s), terminating the process...")
        p.terminate()
        p.join()
        result = None
    else:
        print(f"{datetime.now()} [INFO] Function {str(func)} executed successfully")
        result = q.get()
    return result

def generate_diff_html(text1, text2):
    diff = unified_diff(text1.splitlines(keepends=True),
                        text2.splitlines(keepends=True),
                        fromfile='text1', tofile='text2')

    diff_html = ""
    for line in diff:
        if line.startswith('+'):
            diff_html += f"<div style='color:green;'>{line.rstrip()}</div>"
        elif line.startswith('-'):
            diff_html += f"<div style='color:red;'>{line.rstrip()}</div>"
        elif line.startswith('@'):
            diff_html += f"<div style='color:blue;'>{line.rstrip()}</div>"
        else:
            diff_html += f"{line.rstrip()}<br>"
    return diff_html


newline_char = '\n'


def RAG(question, draft_paragraphs):
    answer = draft_paragraphs
    print(f"{datetime.now()} [INFO] Modifying the draft...")
    print("0" * 80)

    print(f"{datetime.now()} [INFO] Generating corresponding query...")
    res = run_with_timeout(get_query_wrapper, 3, question, answer)
    if not res:
        print(f"{datetime.now()} [INFO] Skipping subsequent steps...")
        return answer
    else:
        query = res

    print(f">>> Query: {query.replace('\n', ' ')}")
    print(f"{datetime.now()} [INFO] Retrieving database document...")
    res = run_with_timeout(get_content_wrapper, 5, query)
    if not res:
        print(f"{datetime.now()} [INFO] Skipping subsequent steps...")
        return answer
    else:
        content = res

    for j, c in enumerate(content):
        print(f"{datetime.now()} [INFO] Modifying the answer based on database document...[{j+1}/{min(len(content), 3)}]")
        res = run_with_timeout(get_revise_answer_wrapper, 10, question, answer, c)
        if not res:
            print(f"{datetime.now()} [INFO] Skipping subsequent steps...")
            continue
        else:
            diff_html = generate_diff_html(answer, res)
            display(HTML(diff_html))
            answer = res

        print(f"{datetime.now()} [INFO] Answer modification complete [{j}/{min(len(content), 3)}]")

    return answer

def process_with_agents(question, previous_answer, num_agent):
    all_agent_results = []
    for agent in range(num_agent):
        print(f"{datetime.now()} [INFO] Agent {agent + 1} is obtaining the Step draft...")
        checking_instruction = get_checking_draft_cot(question, previous_answer)
        print(f"{datetime.now()} [INFO] Agent {agent + 1} is processing the draft...")
        draft_paragraphs = get_revise_draft_cot(question, previous_answer, checking_instruction)

        print(f"##################### DRAFT #######################")
        print(draft_paragraphs)
        print(f"#####################  END  #######################")

        print(f"{datetime.now()} [INFO] Agent {agent + 1} is processing the draft using RAG...")
        answer_first_state = RAG(question, draft_paragraphs)

        # Put the result of the agent into the list
        all_agent_results.append(answer_first_state)


    integrated_answer = integrate_agents_answers(question, all_agent_results)
    return integrated_answer


def ratt(question, args):
    step_num = args.num_steps
    all_agent_results = []

    print(f"{datetime.now()} [INFO] Start to get Step 1 draft...")

    for agent in range(args.num_agents):
        print(f"{datetime.now()} [INFO] Agent {agent + 1} is obtaining the Step draft...")
        draft = get_draft_cot(question)

        print(f"{datetime.now()} [INFO] Agent {agent + 1} is processing the draft...")
        draft_paragraphs = generate_code(question, draft)
        print(f"##################### DRAFT #######################")
        print(draft_paragraphs)
        print(f"#####################  END  #######################")

        print(f"{datetime.now()} [INFO] Agent {agent + 1} is processing the draft using RAG...")
        answer_first_state = RAG(question, draft_paragraphs)

        # Put the result of the agent into the list
        all_agent_results.append(answer_first_state)


    answer_intergrate = integrate_agents_answers(question, all_agent_results)
    previous_answer = answer_intergrate

    for iteration in range(1, step_num):
        print(f"{datetime.now()} [INFO] Getting Step {iteration + 1} draft...")

        intergrated_answer = process_with_agents(question, previous_answer, args.num_agents)

        previous_answer = intergrated_answer

        print(f"{datetime.now()} [INFO] Final answer for Step {iteration + 1}: {previous_answer}")

    return previous_answer



def setup_parser():
    # Create a parser
    parser = argparse.ArgumentParser(description="Generate parameters for agent-based iterative output generation.")

    # Add arguments
    parser.add_argument('--num_agents', default=3, type=int, help='Number of agents used for generating outputs simultaneously.')
    parser.add_argument('--num_steps', default=3,type=int, help='Number of iterative steps to run the generation process.')
    parser.add_argument('--final_output_mode', type=str, default='only_last_step', choices=['combine_each_step', 'only_last_step'],
                        help='Method to generate the final output: "combine_each_step" to integrate outputs from each step, "only_last_step" to use the output from the final step as the final output.')

    return parser



if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    num_samples_per_task = 1

    # Printing out the values for demonstration
    print("Number of Agents:", args.num_agents)
    print("Number of Steps:", args.num_steps)
    print("Final Output Mode:", args.final_output_mode)
    print("Number of Samples per Task:", num_samples_per_task)

    problems = read_problems()

    # Get the directory of the current script file
    current_directory = os.path.dirname(__file__)

    # Construct the path to the target directory
    target_directory = os.path.join(current_directory, 'code_evaluation')

    # Ensure the target directory exists, if not, create it
    os.makedirs(target_directory, exist_ok=True)

    # Construct the final file path
    file_path = os.path.join(target_directory, 'samples.jsonl')

    samples = [
        dict(task_id=task_id, completion=ratt(problems[task_id]["prompt"], args))
        for task_id in problems
        for _ in range(num_samples_per_task)
    ]





