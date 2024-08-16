import streamlit as st
import requests
import os

def main():
    st.title('Download Files from URL')

    # Input fields for URL and output path
    url = st.text_input('Enter URL to download:')
    output_path = st.text_input('Enter Output Path (e.g., /path/to/save/):')

    # Download button
    if st.button('Download'):
        if not url:
            st.warning('Please enter a valid URL.')
        elif not output_path:
            st.warning('Please enter a valid output path.')
        else:
            download_file(url, output_path)

def download_file(url, output_path):
    try:
        # Send HTTP GET request to download the file
        response = requests.get(url, stream=True)

        # Check if request was successful
        if response.status_code == 200:
            # Extract filename from URL
            filename = url.split('/')[-1]

            # Create full output path with filename
            file_path = os.path.join(output_path, filename)

            # Save the file to the specified output path
            with open(file_path, 'wb') as f:
                f.write(response.content)

            # Display success message
            st.success(f'File downloaded successfully to: {file_path}')

            # Display download link for the saved file
            st.markdown(get_download_link(file_path), unsafe_allow_html=True)
        else:
            st.error(f'Error downloading file. Status code: {response.status_code}')

    except Exception as e:
        st.error(f'Error downloading file: {str(e)}')

def get_download_link(file_path):
    """Generate a download link for a file."""
    href = f'<a href="file://{os.path.abspath(file_path)}" download>Click here to download</a>'
    return href

if __name__ == '__main__':
    main()
