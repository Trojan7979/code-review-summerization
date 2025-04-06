import requests
from urllib.parse import urljoin, urlparse
from PyPDF2 import PdfReader
import os
import tiktoken
import nltk
from nltk.corpus import stopwords
import re
import nbformat
from nbconvert import PythonExporter
from rich import print
import xml.etree.ElementTree as ET

EXCLUDED_DIRS = ["dist", "node_modules", ".git", "__pycache__", "README.md"]  # Add any other directories to exclude here

def safe_file_read(filepath, fallback_encoding='latin1'):
    try:
        with open(filepath, "r", encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        with open(filepath, "r", encoding=fallback_encoding) as file:
            return file.read()

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

TOKEN = ""

headers = ""

def download_file(url, target_path):
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    with open(target_path, "wb") as f:
        f.write(response.content)

def process_ipynb_file(temp_file):
    with open(temp_file, "r", encoding='utf-8', errors='ignore') as f:
        notebook_content = f.read()

    exporter = PythonExporter()
    python_code, _ = exporter.from_notebook_node(nbformat.reads(notebook_content, as_version=4))
    return python_code

def process_directory(url, output):
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    files = response.json()

    for file in files:
        if file["type"] == "dir" and file["name"] in EXCLUDED_DIRS:
            continue  # Skip excluded directories

        if file["type"] == "file" and is_allowed_filetype(file["name"]):
            print(f"Processing {file['path']}...")

            temp_file = f"temp_{file['name']}"
            download_file(file["download_url"], temp_file)

            output.write(f"# {'-' * 3}\n")
            output.write(f"# Filename: {file['path']}\n")
            output.write(f"# {'-' * 3}\n\n")

            if file["name"].endswith(".ipynb"):
                output.write(process_ipynb_file(temp_file))
            else:
                with open(temp_file, "r", encoding='utf-8', errors='ignore') as f:
                    output.write(f.read())

            output.write("\n\n")
            os.remove(temp_file)
        elif file["type"] == "dir":
            process_directory(file["url"], output)

def process_local_directory(local_path, output):
    for root, dirs, files in os.walk(local_path):
        # Modify dirs in-place to exclude specified directories
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

        for file in files:
            if is_allowed_filetype(file):
                print(f"Processing {os.path.join(root, file)}...")

                output.write(f"# {'-' * 3}\n")
                output.write(f"# Filename: {os.path.join(root, file)}\n")
                output.write(f"# {'-' * 3}\n\n")

                file_path = os.path.join(root, file)

                if file.endswith(".ipynb"):
                    output.write(process_ipynb_file(file_path))
                else:
                    with open(file_path, "r", encoding='utf-8', errors='ignore') as f:
                        output.write(f.read())

                output.write("\n\n")

def process_github_repo(repo_url):
    api_base_url = "https://api.github.com/repos/"
    repo_url_parts = repo_url.split("https://github.com/")[-1].split("/")
    repo_name = "/".join(repo_url_parts[:2])

    # Detect if we have a branch or tag reference
    branch_or_tag = ""
    subdirectory = ""
    if len(repo_url_parts) > 2 and repo_url_parts[2] == "tree":
        # The branch or tag name should be at index 3
        if len(repo_url_parts) > 3:
            branch_or_tag = repo_url_parts[3]
        # Any remaining parts after the branch/tag name form the subdirectory
        if len(repo_url_parts) > 4:
            subdirectory = "/".join(repo_url_parts[4:])
    
    contents_url = f"{api_base_url}{repo_name}/contents"
    contributors_url = f"{api_base_url}{repo_name}/contributors"

    if subdirectory:
        contents_url = f"{contents_url}/{subdirectory}"
    if branch_or_tag:
        contents_url = f"{contents_url}?ref={branch_or_tag}"

    repo_content = [f'<source type="github_repository" url="{repo_url}">']
    contributors_response = get_contributors(contributors_url)

    def process_directory(url, repo_content):
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        files = response.json()

        for file in files:
            if file["type"] == "dir" and file["name"] in EXCLUDED_DIRS:
                continue

            if file["type"] == "file" and is_allowed_filetype(file["name"]):

                print(f"Processing {file['path']}...")

                temp_file = f"temp_{file['name']}"
                download_file(file["download_url"], temp_file)

                repo_content.append(f'<file name="{escape_xml(file["path"])}">') 
                if file["name"].endswith(".ipynb"):
                    repo_content.append(escape_xml(process_ipynb_file(temp_file)))
                else:
                    with open(temp_file, "r", encoding='utf-8', errors='ignore') as f:
                        repo_content.append(escape_xml(f.read()))
                repo_content.append('</file>')
                os.remove(temp_file)

            elif file["type"] == "dir":
                process_directory(file["url"], repo_content)

    process_directory(contents_url, repo_content)
    repo_content.append('</source>')
    print("All files processed.")

    return "\n".join(repo_content), contributors_response

def extract_links(input_file, output_file):
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()
        urls = re.findall(url_pattern, content)
    
    with open(output_file, 'w', encoding='utf-8') as output:
        for url in urls:
            output.write(url + '\n')

def preprocess_text(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as input_file:
        input_text = input_file.read()

    def process_text(text):
        text = re.sub(r"[\n\r]+", "\n", text)
        text = re.sub(r"[^a-zA-Z0-9\s_.,!?:;@#$%^&*()+\-=[\]{}|\\<>`~'\"/]+", "", text)
        text = re.sub(r"\s+", " ", text)
        text = text.lower()
        words = text.split()
        words = [word for word in words if word not in stop_words]
        return " ".join(words)

    try:
        # Parsing the input as XML
        root = ET.fromstring(input_text)

        # Process text content while preserving XML structure
        for elem in root.iter():
            if elem.text:
                elem.text = process_text(elem.text)
            if elem.tail:
                elem.tail = process_text(elem.tail)

        # The processed XML to the output file
        tree = ET.ElementTree(root)
        tree.write(output_file, encoding="utf-8", xml_declaration=True)
        print("Text preprocessing completed with XML structure preserved.")
    except ET.ParseError:
        # If XML parsing fails, process the text without preserving XML structure
        processed_text = process_text(input_text)
        with open(output_file, "w", encoding="utf-8") as out_file:
            out_file.write(processed_text)
        print("XML parsing failed. Text preprocessing completed without XML structure.")

def get_token_count(text, disallowed_special=[], chunk_size=1000):
    enc = tiktoken.get_encoding("cl100k_base")

    # Remove XML tags
    text_without_tags = re.sub(r'<[^>]+>', '', text)

    # Split the text into smaller chunks
    chunks = [text_without_tags[i:i+chunk_size] for i in range(0, len(text_without_tags), chunk_size)]
    total_tokens = 0

    for chunk in chunks:
        tokens = enc.encode(chunk, disallowed_special=disallowed_special)
        total_tokens += len(tokens)
    
    return total_tokens

def is_same_domain(base_url, new_url):
    return urlparse(base_url).netloc == urlparse(new_url).netloc

def is_within_depth(base_url, current_url, max_depth):
    base_parts = urlparse(base_url).path.rstrip('/').split('/')
    current_parts = urlparse(current_url).path.rstrip('/').split('/')

    if current_parts[:len(base_parts)] != base_parts:
        return False

    return len(current_parts) - len(base_parts) <= max_depth

def process_pdf(url):
    response = requests.get(url)
    response.raise_for_status()

    with open('temp.pdf', 'wb') as pdf_file:
        pdf_file.write(response.content)

    text = []
    with open('temp.pdf', 'rb') as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        for page in range(len(pdf_reader.pages)):
            text.append(pdf_reader.pages[page].extract_text())

    os.remove('temp.pdf')
    return ' '.join(text)
        
def process_github_pull_request(pull_request_url):
    url_parts = pull_request_url.split("/")
    repo_owner = url_parts[3]
    repo_name = url_parts[4]
    pull_request_number = url_parts[-1]

    api_base_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls/{pull_request_number}"
    headers = {"Authorization": f"token {TOKEN}"}

    response = requests.get(api_base_url, headers=headers)
    pull_request_data = response.json()

    diff_url = pull_request_data["diff_url"]
    diff_response = requests.get(diff_url, headers=headers)
    pull_request_diff = diff_response.text

    comments_url = pull_request_data["comments_url"]
    review_comments_url = pull_request_data["review_comments_url"]
    comments_response = requests.get(comments_url, headers=headers)
    review_comments_response = requests.get(review_comments_url, headers=headers)
    comments_data = comments_response.json()
    review_comments_data = review_comments_response.json()

    all_comments = comments_data + review_comments_data
    all_comments.sort(key=lambda comment: comment.get("position") or float("inf"))

    formatted_text = f'<source type="github_pull_request" url="{pull_request_url}">\n'
    formatted_text += '<pull_request_info>\n'
    formatted_text += f'<title>{escape_xml(pull_request_data["title"])}</title>\n'
    formatted_text += f'<description>{escape_xml(pull_request_data["body"])}</description>\n'
    formatted_text += '<merge_details>\n'
    formatted_text += f'{escape_xml(pull_request_data["user"]["login"])} wants to merge {pull_request_data["commits"]} commit into {repo_owner}:{pull_request_data["base"]["ref"]} from {pull_request_data["head"]["label"]}\n'
    formatted_text += '</merge_details>\n'
    formatted_text += '<diff_and_comments>\n'

    diff_lines = pull_request_diff.split("\n")
    comment_index = 0
    for line in diff_lines:
        formatted_text += f'{escape_xml(line)}\n'
        while comment_index < len(all_comments) and all_comments[comment_index].get("position") == diff_lines.index(line):
            comment = all_comments[comment_index]
            formatted_text += f'<review_comment>\n'
            formatted_text += f'<author>{escape_xml(comment["user"]["login"])}</author>\n'
            formatted_text += f'<content>{escape_xml(comment["body"])}</content>\n'
            formatted_text += f'<path>{escape_xml(comment["path"])}</path>\n'
            formatted_text += f'<line>{comment["original_line"]}</line>\n'
            formatted_text += '</review_comment>\n'
            comment_index += 1

    formatted_text += '</diff_and_comments>\n'
    formatted_text += '</pull_request_info>\n'

    repo_url = f"https://github.com/{repo_owner}/{repo_name}"
    repo_content = process_github_repo(repo_url)
    
    formatted_text += '<repository>\n'
    formatted_text += repo_content
    formatted_text += '</repository>\n'
    formatted_text += '</source>'

    print(f"Pull request {pull_request_number} and repository content processed successfully.")

    return formatted_text
    
def escape_xml(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        # Remove the following lines to stop converting apostrophes and quotes
        # .replace("\"", "&quot;")
        # .replace("'", "&apos;")
    )

def process_github_issue(issue_url):
    url_parts = issue_url.split("/")
    repo_owner = url_parts[3]
    repo_name = url_parts[4]
    issue_number = url_parts[-1]

    api_base_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues/{issue_number}"
    headers = {"Authorization": f"token {TOKEN}"}

    response = requests.get(api_base_url, headers=headers)
    issue_data = response.json()

    comments_url = issue_data["comments_url"]
    comments_response = requests.get(comments_url, headers=headers)
    comments_data = comments_response.json()

    formatted_text = f'<source type="github_issue" url="{issue_url}">\n'
    formatted_text += '<issue_info>\n'
    formatted_text += f'<title>{escape_xml(issue_data["title"])}</title>\n'
    formatted_text += f'<description>{escape_xml(issue_data["body"])}</description>\n'
    formatted_text += '<comments>\n'

    for comment in comments_data:
        formatted_text += '<comment>\n'
        formatted_text += f'<author>{escape_xml(comment["user"]["login"])}</author>\n'
        formatted_text += f'<content>{escape_xml(comment["body"])}</content>\n'

        code_snippets = re.findall(r'https://github.com/.*#L\d+-L\d+', comment['body'])
        for snippet_url in code_snippets:
            url_parts = snippet_url.split("#")
            file_url = url_parts[0].replace("/blob/", "/raw/")
            line_range = url_parts[1]
            start_line, end_line = map(int, line_range.split("-")[0][1:]), map(int, line_range.split("-")[1][1:])

            file_response = requests.get(file_url, headers=headers)
            file_content = file_response.text

            code_lines = file_content.split("\n")[start_line-1:end_line]
            code_snippet = "\n".join(code_lines)

            formatted_text += '<code_snippet>\n'
            formatted_text += f'<![CDATA[{code_snippet}]]>\n'
            formatted_text += '</code_snippet>\n'

        formatted_text += '</comment>\n'

    formatted_text += '</comments>\n'
    formatted_text += '</issue_info>\n'

    repo_url = f"https://github.com/{repo_owner}/{repo_name}"
    repo_content = process_github_repo(repo_url)
    
    formatted_text += '<repository>\n'
    formatted_text += repo_content
    formatted_text += '</repository>\n'
    formatted_text += '</source>'

    print(f"Issue {issue_number} and repository content processed successfully.")

    return formatted_text


def is_excluded_file(filename):
    """
    Check if a file should be excluded based on patterns.

    Args:
        filename (str): The file path to check

    Returns:
        bool: True if the file should be excluded, False otherwise
    """
    excluded_patterns = [
        '.pb.go',  # Proto generated Go files
        '_grpc.pb.go',  # gRPC generated Go files
        'mock_',  # Mock files
        '/generated/',  # Generated files in a generated directory
        '/mocks/',  # Mock files in a mocks directory
        '.gen.',  # Generated files with .gen. in name
        '_generated.',  # Generated files with _generated in name
    ]

    return any(pattern in filename for pattern in excluded_patterns)


def is_allowed_filetype(filename):
    """
    Check if a file should be processed based on its extension and exclusion patterns.

    Args:
        filename (str): The file path to check

    Returns:
        bool: True if the file should be processed, False otherwise
    """
    # First check if file matches any exclusion patterns
    if is_excluded_file(filename):
        return False

    # Then check if it has an allowed extension
    allowed_extensions = [
        '.go',
        '.proto',
        '.py',
        '.txt',
        '.md',
        '.cjs',
        '.html',
        '.json',
        '.ipynb',
        '.h',
        '.localhost',
        '.yaml',
        '.example'
    ]

    return any(filename.endswith(ext) for ext in allowed_extensions)

def get_contributors(url):

    import requests
    response = requests.request("GET", url, headers=headers)
    
    return response.text

def get_github_content(url, token):
        
        try:
            global TOKEN
            global headers
            TOKEN = token
            headers = {"Authorization": f"Bearer {token}", "Accept" : "application/vnd.github+json", "X-GitHub-Api-Version" : "2022-11-28"}
        except Exception as e:
            print(e)

        try:
            if "github.com" in url:
                if "/pull/" in url:
                    final_output = process_github_pull_request(url)
                elif "/issues/" in url:
                    final_output = process_github_issue(url)
                else:
                    final_output, contributors_response = process_github_repo(url)
                    
            return final_output, contributors_response
        
        except Exception as e:
            raise  # Re-raise the exception for debugging purposes

if __name__ == "__main__":
    res, res1 = get_github_content("https://github.com/Trojan7979/html-portfolio", "")
    print(res)