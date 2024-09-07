import requests

from infer import logger


def download_file(url, max_size_bytes, output_filename, api_key=None):
    """
    Download a file from a given URL with size limit and optional API key.

    Args:
    url (str): The URL of the file to download.
    max_size_bytes (int): Maximum allowed file size in bytes.
    output_filename (str): The name of the file to save the download as.
    api_key (str, optional): API key to be used as a bearer token.

    Returns:
    bool: True if download was successful, False otherwise.
    """
    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad requests

        # Get the file size if possible
        file_size = int(response.headers.get("Content-Length", 0))

        if file_size > max_size_bytes:
            print(
                f"File size ({file_size} bytes) exceeds the maximum allowed size ({max_size_bytes} bytes)."
            )
            return False

        # Download and write the file
        downloaded_size = 0
        with open(output_filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                downloaded_size += len(chunk)
                if downloaded_size > max_size_bytes:
                    print(
                        f"Download stopped: Size limit exceeded ({max_size_bytes} bytes)."
                    )
                    return False
                file.write(chunk)

        logger.info(f"File downloaded successfully: {output_filename}")
        return True

    except requests.RequestException as e:
        print(f"Error downloading file: {e}")
        return False
