import streamlit as st
import boto3
import os
import requests
import threading

# Function to monitor EC2 instance status
def monitor_ec2_instance(instance_id):
    try:
        ec2 = boto3.resource('ec2')
        instance = ec2.Instance(instance_id)
        
        # Get current instance status
        status = instance.state['Name']
        
        return status

    except Exception as e:
        st.error(f'Error monitoring EC2 instance: {str(e)}')
        return None

# Function to download model from EC2 instance
def download_model_from_ec2(url, output_path):
    try:
        # Send HTTP GET request to download the model file
        response = requests.get(url, stream=True)
        
        # Check if request was successful
        if response.status_code == 200:
            # Create full output path with filename
            filename = url.split('/')[-1]
            file_path = os.path.join(output_path, filename)
            
            # Save the file to the specified output path
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            
            # Display success message
            st.success(f'Model downloaded successfully to: {file_path}')

        else:
            st.error(f'Error downloading model. Status code: {response.status_code}')

    except Exception as e:
        st.error(f'Error downloading model from EC2: {str(e)}')

def main():
    st.title('EC2 Instance Monitor and Model Downloader')

    # Input fields for EC2 instance ID, URL, and output path
    ec2_instance_id = st.text_input('Enter EC2 Instance ID:')
    model_url = st.text_input('Enter Model URL from EC2:')
    output_path = st.text_input('Enter Output Path (e.g., /path/to/save/):')

    # Button to monitor EC2 instance status
    if st.button('Monitor EC2 Instance Status'):
        if not ec2_instance_id:
            st.warning('Please enter a valid EC2 instance ID.')
        else:
            status = monitor_ec2_instance(ec2_instance_id)
            if status:
                st.success(f'EC2 Instance Status: {status}')
    
    # Button to download model from EC2
    if st.button('Download Model from EC2'):
        if not model_url:
            st.warning('Please enter a valid Model URL from EC2.')
        elif not output_path:
            st.warning('Please enter a valid output path.')
        else:
            # Start downloading model from EC2 in a separate thread
            threading.Thread(target=download_model_from_ec2, args=(model_url, output_path)).start()

if __name__ == '__main__':
    main()
