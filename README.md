# RAG-Google-Doc  
A Retrieval-Augmented Generation (RAG) system using LlamaIndex and Google Drive.

---

## 1. Getting Access to Google Drive and Google Docs

### Step 1: Create a Google Cloud Project
1. Log in to the [Google Cloud Console](https://console.cloud.google.com/).
2. Sign in with the Google account where your Google Docs are stored.
3. Click on **Create Project** and provide a suitable name for your project.
4. Note the **Project ID** for future reference.

### Step 2: Enable Required APIs
1. Navigate to **APIs & Services > Library**.
2. Search for **Google Drive API** and **Google Docs API**, then enable them.
3. Ensure the following APIs are enabled:
   - **Google Drive API**  
   - **Google Docs API**  

### Step 3: Create OAuth 2.0 Credentials
1. Go to **APIs & Services > Credentials**.
2. Click **Create Credentials** and select **OAuth 2.0 Client IDs**.
3. **Configure the OAuth Consent Screen** (if prompted):
   - Choose **External** for the user type.
   - Fill in the required fields (e.g., **App Name**, **User Support Email**).
   - Save and continue without adding scopes (the default email scope is sufficient for personal use).
   - Add your email under the **Test Users** section and finish.
4. **Choose Application Type**:
   - Select **Desktop App**.
   - Name it (e.g., `"RAG System App"`) and click **Create**.
5. **Download `credentials.json`**:
   - After creating the credentials, click **Download** and save the file as `credentials.json`.

### Step 4: Add Yourself as a Tester
1. Navigate to **OAuth Consent Screen**.
2. Scroll to the **Test Users** section and add your Google account.
3. Publish the app for testing purposes.

---

## 2. Code Usage

### **Set Up API Keys**
1. Create a file **`load_keys.py`** where all the API keys can be loaded. Example implementation:
    ```python
    import os
    
    def load_openai_key():
        os.environ['OPENAI_API_KEY'] = "your-openai-api-key"
    ```
2. **Load the keys:** The main file is `rag_doc_2.py`. Open it and load your keys.

---

### **Creating the Index**
1. **Instantiate `RagGoogleDoc`**, specifying the **IDs of folders or files** to be added to the vector index.  
   - **Note:** Only Google Docs will be indexed.
2. **Use the `create_index()` method** to build the index.

#### **Example Usage:**
```python
from rag_google_doc import RagGoogleDoc

google_drive_folder_ids = [
    '14rNXFJe-WnK3AyIoDSX_8zNCVVFiOXed',
    '12HEHe7876pCtuL5Z4makI-5E5cd2fLx4',
    '1US5wqXvJYw6u98yCfxsr9E42lmoaSDyp'
]

# Initialize the RAG system
rag_obj = RagGoogleDoc(google_drive_folder_ids, save_index_address='Data/google_db')

# Create the index
db = rag_obj.create_index()

```

### Query Index
1. Create your query.
2. Use the method `query_index(query)` method to retrieve relevant information.
```commandline
    query = 'What algorithm of salient object detection can be implemented in keras more quickly?'
    response = rag_obj.query_index(query)
    
    print(response)
```

## 3 Notes
1. You can extend this system to support PDFs, TXT files, or other document types.
2. Feel free to explore alternative vector stores like ChromaDB or Weaviate.

Let me know if you have any questions or improvements!