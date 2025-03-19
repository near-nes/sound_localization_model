ciao desa
questo tutorial dovrebbe guidarti su come creare un bucket per storage su google cloud e usarlo per metterci i risultati delle simulazioni. ho scelto google cloud perché ha una free trial GENEROSA (300$ per tre mesi).
LEGGIMI: ti ho allegato le mie credenziali a google cloud. se sei di corsa, puoi usarle, ma mandami un messaggio di check-in per dirmelo e facci un minimo attenzione. queste credenziali hanno diritto sugli object storage di google: la cosa peggiore che possa succedere è che qualcuno (apposta) fa di tutto per farmi finire i miei crediti gratis (non preoccuparti, praticamente impossibile se non lo fai apposta), o che sono usate dopo la scadenza: May 23, 2025

# Storage Bucket

## create storage bucket
- Go to [Google Cloud Console](https://console.cloud.google.com)
- Create a new project (e.g., "SimulationProject")
- Navigate to Cloud Storage → Buckets
- Click "Create" → Name your bucket (e.g., "sim-results-2024")
- Choose region → Click "Create" (i used zurich, single region)

## create service account for authentication
- Go to IAM & Admin → Service Accounts (use search function on top of page)
- Click "Create Service Account"
- Name it "simulation-uploader"
- Under "Grant Access", add role: Storage Object Admin
- Click "Done" → Go to the service account → "Keys" tab
- Click "Add Key → Create new key → JSON" → Download the JSON file\

# test on you laptop 

## upload test on your laptop
- create a new folder
- create a virtualenv to have a dedicated python environment: `python -m venv venv`. if you prefer using the existing one (doesn't change much) just skip this and the next step.
- activate the environment: `source venv/bin/activate`
- install required packages: `pip install google-cloud-storage ipykernel`
- open code
- remember to select the correct environment (the one in `venv/bin`): for python files, select it in the bottom right of the lower status bar; for ipython notebooks, select it at the top of the file. 
- run the upload test: `python upload_sim_res.py`

## check uploaded files, delete some, delete all
- open `list_and_download.ipynb`
- run the function you want

# upload and run on server

- upload the whole `.zip`
- read the `desa_generate_results.py` i included. As you can see, i made three small changes:
    1. imported necessary files `from upload_sim_res import upload_to_gcs`
    2. included variable to control behavior `UPLOAD_AND_DELETE = False`
    3. uploaded AND LOGGED.
```py
if UPLOAD_AND_DELETE:
    logger.warning(f"uploading {result_file} to GCS...")
    # upload results to GCS
    upload_to_gcs(result_file)
    logger.warning(f"uploaded {result_file} to GCS. Deleting local file...")
    # delete local file
    result_file.unlink()
    logger.info(f"deleted {result_file}")
```
