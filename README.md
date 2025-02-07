# rag-google-doc
RAG system using LlamaIndex and google drive.


## 1. Get access to google drive and google docs

### Step 1: Create a Google Cloud Project
1. Log in to the Google Cloud Console.
2. Sign in with the google account where the google docs are stored.
3. Click on Create Project and provide a suitable name for your project.
4. Note the Project ID for future reference.

### Step 2: Enable Required APIs
1. Navigate to APIs & Services > Library.
2. Search for "Google Drive API" and "Google Docs API." Click each API and enable them.
3. Enable the following APIs:
   * Google Drive API 
   * Google Docs API

### Step 3: Create OAuth 2.0 Credentials
1. Go to Credentials:
   * Navigate to APIs & Services > Credentials in the left-hand menu.
2. Create Credentials:
   * Click the Create Credentials button at the top.
   * Select OAuth 2.0 Client IDs.
3. Configure Consent Screen:
If prompted, configure the OAuth consent screen:
   * Choose "External" for the user type.
   * Fill in the required fields (e.g., App Name, User Support Email).
   * Save and continue without adding scopes (the default email scope is sufficient for personal use).
   * Add your email under the "Test Users" section and finish.

4. Choose Application Type:
   * Select "Desktop App" for the application type.
   * Name it (e.g., "RAG System App") and click "Create."

5. Download credentials.json:
   * After creating the credentials, click the Download button to save the credentials.json file.

### Step 4: Add Yourself as a Tester
1. Navigate to OAuth Consent Screen.
2. Add your Google account as a test user. Scroll to the Test Users section. Add your email address (the one you're using to log into Google Drive).
3. Publish the app for testing purposes.

## 2. 