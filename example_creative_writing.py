# Basic Tool Functions
import os
os.environ["OPENAI_API_KEY"] = "" # Your Openai API key
os.environ["GOOGLE_CSE_ID"] = "" # Your Google CSE ID
os.environ["GOOGLE_API_KEY"] = "" # Your Google API Key

from openai import OpenAI
from langchain.tools import Tool
# from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import AsyncHtmlLoader
from datetime import datetime
from multiprocessing import Process, Queue
from difflib import unified_diff
from IPython.display import display, HTML
import argparse
import tiktoken

api_key = "" # Your Openai API key
openai_client = OpenAI(api_key=api_key)



# RATT Pipeline
chatgpt_system_prompt = f'''
You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.
Knowledge cutoff: 2023-04
Current date: {datetime.now().strftime('%Y-%m-%d')}
'''

def get_search(query:str="", k:int=1): # get the top-k resources with google
    search = GoogleSearchAPIWrapper(k=k)
    def search_results(query):
        return search.results(query, k)
    tool = Tool(
        name="Google Search Snippets",
        description="Search Google for recent results.",
        func=search_results,
    )
    ref_text = tool.run(query)
    if 'Result' not in ref_text[0].keys():
        return ref_text
    else:
        return None


def get_page_content(link:str):
    loader = AsyncHtmlLoader([link])
    docs = loader.load()
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    if len(docs_transformed) > 0:
        return docs_transformed[0].page_content
    else:
        return None


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
    # Add sentence by sentence to the text chunk, ensuring each chunk is less than 2k tokens
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
            # Ensure that the extra characters are distributed among the first extra parts
            end = start + part_length + (1 if i < extra else 0)
            parts.append(text[start:end])
            start = end
        return parts



def get_draft_tot_inital(question, num_agents=3):

    draft_prompt = '''
IMPORTANT:
Try to answer this question/instruction with step-by-step thoughts and make the answer more structural.
Use `\n\n` to split the answer into several paragraphs.
Just respond to the instruction directly. DO NOT add additional explanations or introducement in the answer unless you are asked to.
'''

    refine_prompt = '''
Referencing the answers provided by all agents, synthesize a more detailed and comprehensive response by integrating all relevant details from these answers. Ensure logical coherence and provide ONLY THE MERGED ANSWER AS THE OUTPUT, omitting any discussion of the comparison process or analytical thoughts.
'''

    agents_drafts = []

    # Loop to generate initial different responses
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

        print(f"{datetime.now()} [INFO] Processing draft...")
        draft_paragraphs = split_draft(draft)
        print(f"{datetime.now()} [INFO] Draft split into {len(draft_paragraphs)} parts")

        # Modify using RAG
        draft_modified = RAG(question, draft_paragraphs)

        # Add each generated draft to the list
        agents_drafts.append(f"Agent{i+1}: {draft_modified}")

        print(f"{datetime.now()} [INFO] Agent{i + 1}/{num_agents} retrieved draft...")




    # Integrate and process previous responses
    agents_input = '\n\n'.join(agents_drafts) + '\n\n' + refine_prompt

    # Generate the integrated answer
    final_draft = openai_client.chat.completions.create(
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

    print(f"{datetime.now()} [INFO] Retrieved integrated draft...")

    return final_draft

def get_draft_tot(question, previous_answer, num_agents=3):
    # Update the draft answer prompt to include the question and previous answer
    draft_prompt = f'''
Base your response on the provided question and the previous answer. Expand the answer by adding more details to enhance its comprehensiveness. Ensure that the expansion maintains logical coherence and enriches the details, making the response more thorough and well-structured.
Question: {question}
Previous Answer: {previous_answer}
IMPORTANT:
Answer the full question with step-by-step thoughts and make the answer more structural.
Use `\n\n` to split the answer into several paragraphs.
Just respond to the instruction directly. DO NOT add additional explanations or introducement in the answer unless you are asked to.
'''

    refine_prompt = '''
Referencing the answers provided by all agents, synthesize a more detailed and comprehensive response by integrating all relevant details from these answers. Ensure logical coherence and provide ONLY THE MERGED ANSWER AS THE OUTPUT, omitting any discussion of the comparison process or analytical thoughts.
'''


    agents_drafts = []

    # Loop to generate initial different responses
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

        print(f"{datetime.now()} [INFO] Processing draft...")
        draft_paragraphs = split_draft(draft)
        print(f"{datetime.now()} [INFO] Draft split into {len(draft_paragraphs)} parts")

        # Modify using RAG
        draft_modified = RAG(question, draft_paragraphs)

        # Add each generated draft to the list
        agents_drafts.append(f"Agent{i + 1}: {draft_modified}")

        print(f"{datetime.now()} [INFO] Agent{i + 1}/{num_agents} retrieved draft...")

    # Integrate and process previous responses
    agents_input = '\n\n'.join(agents_drafts) + '\n\n' + refine_prompt

    # Generate the integrated answer
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

    print(f"{datetime.now()} [INFO] Retrieving integrated draft...")

    # Merge the integrated answer with the previous answer, prioritizing the previous answer with supplementary details from the new answer
    revise_prompt = f'''
Based on the original answer and an additional supplementary answer, generate a response that is richer in detail and logically coherent. Review the original answer:
1. If any part of the answer is correct and requires no further details, retain that portion unchanged and output it directly as it is.
2. For parts that may be improved or lack necessary details, enhance them by integrating information from the supplementary answer to make the response more comprehensive and accurate.
3. If you identify any errors within the answers, correct these errors while ensuring that the revised content remains logically coherent.
Original Answer: {previous_answer}
Supplementary Answer: {final_draft_raw}

**IMPORTANT**
Ensure the revised answer maintains a structured format (multiple paragraphs with subtitles) for better clarity. Separate the paragraphs with `\n\n` characters. Output only the enhanced answer directly, without any extra explanations or announcements unless specifically requested.
'''

    final_draft = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": chatgpt_system_prompt
            },
            {
                "role": "user",
                "content": revise_prompt
            }
        ],
        temperature=1.0
    ).choices[0].message.content

    # Return the final merged draft
    return final_draft


def get_draft(question):
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

def split_draft(draft, split_char='\n\n'):
    # split_char: '\n\n'
    draft_paragraphs = draft.split(split_char)
    # print(f"The draft answer has {len(draft_paragraphs)}")
    return draft_paragraphs

def get_query(question, answer):
    query_prompt = '''
I want to verify the content correctness of the given question, especially the last sentences.
Please summarize the content with the corresponding question.
This summarization will be used as a query to search with Bing search engine.
The query should be short but need to be specific to promise Bing can find related knowledge or pages.
You can also use search syntax to make the query short and clear enough for the search engine to find relevant language data.
Try to make the query as relevant as possible to the last few sentences in the content.
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

def get_content(query):
    res = get_search(query, 1)
    if not res:
        print(">>> No good Google Search Result was found")
        return None
    search_results = res[0]
    link = search_results['link']  # title, snippet
    res = get_page_content(link)
    if not res:
        print(f">>> No content was found in {link}")
        return None
    retrieved_text = res
    trunked_texts = chunk_texts(retrieved_text, 1500)
    trunked_texts = [trunked_text.replace('\n', " ") for trunked_text in trunked_texts]
    return trunked_texts

def filter_irrelevant_content(question, content):
    filter_prompt = '''
Please read the following text and extract only the sections that are relevant to the given question. Organize the extracted information coherently, maintaining the structure of multiple paragraphs with subtitles, and split the paragraphs with `\n\n`.
**Question**: {question}
**Text to Filter**: {content}
**Instruction**: Extract only the relevant information related to the question. Keep the structure clear with multiple paragraphs and subtitles. Provide the filtered information directly without additional explanations or commentary.
'''
    filtered_content = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": filter_prompt
            },
            {
                "role": "user",
                "content": f"##Text: {content}\n\n##Question: {question}\n\n##Instruction: Extract relevant information."
            }
        ],
        temperature=0.5
    ).choices[0].message.content
    return filtered_content

def get_revise_answer(question, answer, content):
    revise_prompt = '''
I want to revise the answer according to retrieved related text of the question in WIKI pages.
You need to check whether the answer is correct.
If you find some errors in the answer, revise the answer to make it better.
If you find some necessary details are ignored, add it to make the answer more plausible according to the related text.
If you find that a part of the answer is correct and does not require any additional details, maintain that part of the answer unchanged. Directly output the original content of that part without any modifications.
**IMPORTANT**
Try to keep the structure (multiple paragraphs with its subtitles) in the revised answer and make it more structual for understanding.
Split the paragraphs with `\n\n` characters.
Just output the revised answer directly. DO NOT add additional explanations or annoucement in the revised answer unless you are asked to.
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
    result = get_content(query)
    q.put(result)

def get_revise_answer_wrapper(q, question, answer, content):
    result = get_revise_answer(question, answer, content)
    q.put(result)

def run_with_timeout(func, timeout, *args, **kwargs):
    q = Queue()  # Create a Queue object for interprocess communication
    # Create a process to execute the given function, passing the Queue and other *args, **kwargs as parameters
    p = Process(target=func, args=(q, *args), kwargs=kwargs)
    p.start()
    # Wait for the process to complete or time out
    p.join(timeout)
    if p.is_alive():
        print(f"{datetime.now()} [INFO] Function {str(func)} execution timed out ({timeout}s), terminating process...")
        p.terminate()  # Terminate the process
        p.join()  # Ensure the process has been terminated
        result = None  # In case of a timeout, we do not have a result
    else:
        print(f"{datetime.now()} [INFO] Function {str(func)} completed successfully")
        result = q.get()  # Retrieve the result from the queue
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
    answer = ""
    for i, p in enumerate(draft_paragraphs):
        print(str(i) * 80)
        print(f"{datetime.now()} [INFO] Processing part {i + 1}/{len(draft_paragraphs)}...")
        answer += '\n\n' + p

        print(f"{datetime.now()} [INFO] Generating corresponding query...")
        res = run_with_timeout(get_query_wrapper, 3, question, answer)
        if not res:
            print(f"{datetime.now()} [INFO] Skipping subsequent steps...")
            continue
        else:
            query = res

        print(f">>> {i}/{len(draft_paragraphs)} Query: {query.replace('\n', ' ')}")
        print(f"{datetime.now()} [INFO] Get web page content...")
        res = run_with_timeout(get_content_wrapper, 5, query)
        if not res:
            print(f"{datetime.now()} [INFO] Skipping subsequent steps...")
            continue
        else:
            content = res
            # content = filter_irrelevant_content(query, res)

        for j, c in enumerate(content):
            if j > 2:
                break
            # 过滤冗杂答案
            c_modified = filter_irrelevant_content(query, c)
            print(f"{datetime.now()} [INFO] Modify answer according to web page content...[{j+1}/{min(len(content), 3)}]")
            res = run_with_timeout(get_revise_answer_wrapper, 10, question, answer, c_modified)
            if not res:
                print(f"{datetime.now()} [INFO] Skipping subsequent steps...")
                continue
            else:
                diff_html = generate_diff_html(answer, res)
                display(HTML(diff_html))
                answer = res

            print(f"{datetime.now()} [INFO] Answer modification completed[{j+1}/{min(len(content), 3)}]")

    return answer


def filter_paragraphs(draft_paragraphs, iteration, step_num):
    iteration = iteration+1
    if draft_paragraphs:
        num_elements_to_keep = int(len(draft_paragraphs) * (iteration / step_num))
        draft_paragraphs = draft_paragraphs[:num_elements_to_keep]
    return draft_paragraphs

def ratt(question,args):

    step_num = args.num_steps
    print(f"{datetime.now()} [INFO] Retrieving Step 1 draft...")
    draft = get_draft_tot_inital(question,args.num_agents)
    print(f"{datetime.now()} [INFO] Step 1 draft returned")
    print(f"##################### DRAFT #######################")
    print(draft)
    print(f"#####################  END  #######################")

    print(f"{datetime.now()} [INFO] Processing draft...")
    draft_paragraphs = split_draft(draft)
    print(f"{datetime.now()} [INFO] Draft split into {len(draft_paragraphs)} parts")


    answer_first_state = RAG(question, draft_paragraphs)

    previous_answer = answer_first_state

    each_step_drafts = [f"Step 1 \n: {previous_answer}"]

    for iteration in range(1, step_num):
        print(f"{datetime.now()} [INFO] Retrieving Step {iteration + 1} draft...")
        draft = get_draft_tot(question, previous_answer, num_agents=args.num_agents)
        print(f"{datetime.now()} [INFO] Step {iteration + 1} draft returned")
        print(f"##################### DRAFT #######################")
        print(draft)
        print(f"#####################  END  #######################")

        print(f"{datetime.now()} [INFO] Processing draft...")
        draft_paragraphs = split_draft(draft)
        print(f"{datetime.now()} [INFO] Draft split into {len(draft_paragraphs)} parts")

        # filtered_paragraphs = filter_paragraphs(draft_paragraphs, iteration, step_num)
        final_answer = RAG(question, draft_paragraphs)

        each_step_drafts.append(f"Step {iteration + 1} \n: {final_answer}")

        # Update previous_answer for the current iteration's response
        previous_answer = final_answer

    # Obtain the COT answer for baseline comparison
    draft_cot = get_draft(question)

    if args.final_output_mode == 'combine_each_step':
        final_draft = '\n\n'.join(each_step_drafts)
        refine_prompt = f'''
Referencing the answers provided by each step, synthesize a more detailed and comprehensive response by integrating all relevant details from these answers. Ensure logical coherence and provide ONLY THE MERGED ANSWER AS THE OUTPUT, omitting any discussion of the comparison process or analytical thoughts.
'''
        previous_answer = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": chatgpt_system_prompt
                },
                {
                    "role": "user",
                    "content": final_draft + '\n\n' + refine_prompt
                }
            ],
            temperature=1.0
        ).choices[0].message.content

    return draft_cot, previous_answer

def setup_parser():
    # Create a parser
    parser = argparse.ArgumentParser(description="Generate parameters for agent-based iterative output generation.")

    # Add arguments
    parser.add_argument('--num_agents', default=3, type=int, help='Number of agents used for generating outputs simultaneously.')
    parser.add_argument('--num_steps', default=3,type=int, help='Number of iterative steps to run the generation process.')
    parser.add_argument('--final_output_mode', type=str, default='only_last_step', choices=['combine_each_step', 'only_last_step'],
                        help='Method to generate the final output: "combine_each_step" to integrate outputs from each step, "only_last_step" to use the output from the final step as the final output.')

    return parser

# Example usage
if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    # Printing out the values for demonstration
    print("Number of Agents:", args.num_agents)
    print("Number of Steps:", args.num_steps)
    print("Final Output Mode:", args.final_output_mode)

    draft, answer = ratt("Introduce Jin-Yong's Life.", args)
    diff_html = generate_diff_html(draft, answer)
    with open("diff.html", "w") as file:
        file.write(diff_html)
    # display(HTML(diff_html))

    # Print the draft and answer for review
    print("Draft (Chain of Thought):")
    print(draft)
    print("\nAnswer (RATT):")
    print(answer)
